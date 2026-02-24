"""Run baseline policy rollouts on the multizones_reference_buildings dataset.

For each requested building type, selects the first *n* buildings (by
building_id) from the dataset index so that the selection is deterministic
and reproducible.  Each building goes through the full pipeline
(meters + controllable), an environment is created, and a rollout is
executed with the chosen baseline controller.

Results are streamed to a JSONL file so partial progress is preserved.
"""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from b2b.benchmark.runner import EpisodeResult, PolicyLike, run_rollout
from b2b.env import STORE_PATH
from b2b.pipeline import extract_discovery_metadata
from b2b.simulator import create_simulator
from b2b.sources.multizones_reference_buildings import (
    BuildingType,
    dataset_zip,
    search_buildings,
)
from b2b.store import Constant, ExtractFromZip, realize
from b2b.types import BaseRewardConfig, BuildingConfig

logger = logging.getLogger(__name__)

ALL_BUILDING_TYPES: list[BuildingType] = [
    "Warehouse",
    "HotelSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    building_id: int
    building_type: str
    place: str
    episode_result: EpisodeResult | None
    success: bool
    error: str | None


def _short_error(e: BaseException) -> str:
    msg = str(e).strip().replace("\n", " ")
    if len(msg) > 240:
        msg = msg[:240] + "…"
    return f"{type(e).__name__}: {msg}"


def _load_sb3_policy(policy_cfg: Any) -> PolicyLike:
    """Load a Stable Baselines 3 checkpoint.

    Expected config keys:
        algorithm:       SB3 algorithm name (e.g. "ppo", "sac", "trpo")
        checkpoint_path: path to the saved ``.zip`` model file
    """
    import importlib

    algorithm = str(getattr(policy_cfg, "algorithm", "")).strip()
    checkpoint_path = str(getattr(policy_cfg, "checkpoint_path", "")).strip()
    if not algorithm:
        raise ValueError("policy.algorithm is required for sb3 policies")
    if not checkpoint_path:
        raise ValueError("policy.checkpoint_path is required for sb3 policies")

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SB3 checkpoint not found: {path}")

    algo_upper = algorithm.upper()
    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algorithm}.{algorithm}")
            algo_cls = getattr(module, algo_upper)
            model = algo_cls.load(str(path))
            logger.info("Loaded SB3 %s model from %s", algo_upper, path)
            return model
        except (ModuleNotFoundError, AttributeError):
            continue

    raise ImportError(
        f"SB3 algorithm {algorithm!r} not found in stable_baselines3 or sb3_contrib."
    )


def _load_custom_policy(policy_cfg: Any) -> PolicyLike:
    """Dynamically import and instantiate a custom policy class.

    Expected config keys:
        module:     fully-qualified Python module (e.g. "my_package.policies")
        class_name: class name within that module (e.g. "MyPolicy")
        kwargs:     (optional) dict of keyword arguments passed to the constructor
    """
    import importlib

    module_path = str(getattr(policy_cfg, "module", "")).strip()
    class_name = str(getattr(policy_cfg, "class_name", "")).strip()
    if not module_path or not class_name:
        raise ValueError(
            "policy.module and policy.class_name are required for custom policies"
        )

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    raw_kwargs = getattr(policy_cfg, "kwargs", None)
    if raw_kwargs is not None:
        from omegaconf import OmegaConf as _OC

        kwargs = _OC.to_container(raw_kwargs, resolve=True)
        if not isinstance(kwargs, dict):
            raise TypeError(f"policy.kwargs must be a mapping, got {type(kwargs).__name__}")
    else:
        kwargs = {}

    policy = cls(**kwargs)
    if not hasattr(policy, "predict"):
        raise TypeError(
            f"{module_path}.{class_name} does not expose a predict() method"
        )
    return policy


def make_policy_from_cfg(cfg: DictConfig) -> PolicyLike:
    """Instantiate a policy from a Hydra config group.

    Built-in types:
        zone_temp_21, fan_coil_constant, unitary_pi, unitary_sat,
        unitary_airflow_first_sat, air_loop_sat

    For trained RL agents::

        policy:
          type: sb3
          algorithm: ppo          # any SB3 / sb3-contrib algo
          checkpoint_path: /path/to/best_model.zip

    For arbitrary Python classes::

        policy:
          type: custom
          module: my_package.policies
          class_name: MyPolicy
          kwargs:                  # optional constructor keyword arguments
            target_temp_c: 22.0
    """
    policy_type = str(getattr(cfg.policy, "type", "")).strip()

    if policy_type == "zone_temp_21":
        from b2b.baselines.controllers.zone_temp_21 import ZoneTemp21Policy

        return ZoneTemp21Policy(cfg.policy)
    if policy_type == "fan_coil_constant":
        from b2b.baselines.controllers.fan_coil_constant import FanCoilConstantPolicy

        return FanCoilConstantPolicy(cfg.policy)
    if policy_type == "unitary_pi":
        from b2b.baselines.controllers.unitary_pi import UnitaryPIPolicy

        return UnitaryPIPolicy(cfg.policy)
    if policy_type in ("unitary_sat", "unitary_airflow_first_sat"):
        from b2b.baselines.controllers.unitary_sat import UnitaryAirflowFirstSatPolicy

        return UnitaryAirflowFirstSatPolicy(cfg.policy)
    if policy_type == "air_loop_sat":
        from b2b.baselines.controllers.air_loop_sat import AirLoopSatPolicy

        return AirLoopSatPolicy(cfg.policy)
    if policy_type == "sb3":
        return _load_sb3_policy(cfg.policy)
    if policy_type == "custom":
        return _load_custom_policy(cfg.policy)

    raise NotImplementedError(
        f"Unsupported policy.type={policy_type!r}. "
        "Use a built-in baseline name, 'sb3' with checkpoint_path, "
        "or 'custom' with module/class_name."
    )


def select_buildings(
    building_type: BuildingType,
    n: int,
) -> list[dict[str, Any]]:
    """Return the first *n* buildings of *building_type*, sorted by building_id."""
    df = search_buildings(building_type=building_type)
    df = df.sort_values("building_id").head(n).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(dict(row))
    return rows


def build_config(
    row: dict[str, Any],
    eplus_output_dir: Path,
    energy_weight: float = 0.0,
) -> BuildingConfig:
    """Run the pipeline for a single building and return a BuildingConfig."""
    store = STORE_PATH.get()
    root_zip = dataset_zip()

    control_derivation = row["derivation_thunk"]()
    epjson_path, hvac_equipment = realize(store, control_derivation)

    epw_derivation = ExtractFromZip(root_zip, f"dataset/{row['weather_file']}")
    epw_path = realize(store, epw_derivation)

    metadata = realize(
        store,
        extract_discovery_metadata(Constant(epjson_path), epw_derivation),
    )

    return BuildingConfig(
        path_to_building=epjson_path,
        path_to_weather=epw_path,
        reward_config=BaseRewardConfig(energy_weight=energy_weight),
        hvac_equipment=hvac_equipment,
        eplus_output_dir=eplus_output_dir,
        warmup_phases=metadata.warmup_phases,
        area=metadata.net_conditioned_area,
        source_metadata={
            "source": "multizones_reference_buildings",
            "building_id": int(row["building_id"]),
            "building_type": str(row["building_type"]),
            "place": str(row["place"]),
        },
    )


def run_multizones_rollout(
    cfg: DictConfig,
    *,
    output_dir: Path,
) -> list[RolloutRecord]:
    """Evaluate a baseline policy on multizones_reference_buildings.

    Reads ``multizones.types``, ``multizones.n_per_type``, and
    ``env.max_steps`` from the Hydra config.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict_any = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = cfg_dict_any if isinstance(cfg_dict_any, dict) else {}

    mz = cfg_dict.get("multizones", {})
    mz_dict: dict[str, Any] = mz if isinstance(mz, dict) else {}

    raw_types = mz_dict.get("types", None)
    if raw_types is None or raw_types == []:
        types_to_eval: list[BuildingType] = list(ALL_BUILDING_TYPES)
    else:
        types_to_eval = list(raw_types)

    n_per_type = int(mz_dict.get("n_per_type", 5))

    max_steps_raw = cfg_dict.get("env", {})
    if isinstance(max_steps_raw, dict):
        max_steps_raw = max_steps_raw.get("max_steps", None)
    else:
        max_steps_raw = None
    max_steps = int(max_steps_raw) if max_steps_raw is not None else 365 * 24 * 4

    energy_weight = 0.0
    reward_sect = cfg_dict.get("reward", {})
    if isinstance(reward_sect, dict):
        energy_weight = float(reward_sect.get("energy_weight", 0.0))

    results_path = output_dir / "rollout_results.jsonl"
    errors_dir = output_dir / "errors"

    policy = make_policy_from_cfg(cfg)

    logger.info("Policy: %s", getattr(cfg.policy, "type", "unknown"))
    logger.info("Building types: %s", types_to_eval)
    logger.info("Buildings per type: %d", n_per_type)
    logger.info("Max steps: %d", max_steps)
    logger.info("Output: %s", output_dir)

    records: list[RolloutRecord] = []

    for btype in types_to_eval:
        logger.info("Selecting first %d %s buildings ...", n_per_type, btype)
        rows = select_buildings(btype, n_per_type)
        logger.info("  selected %d buildings", len(rows))

        for idx, row in enumerate(rows, 1):
            bid = int(row["building_id"])
            place = str(row["place"])
            logger.info(
                "[%s %d/%d] building_id=%d  place=%s",
                btype,
                idx,
                len(rows),
                bid,
                place,
            )

            try:
                eplus_dir = output_dir / "eplus_outputs" / str(uuid.uuid4())
                eplus_dir.mkdir(parents=True, exist_ok=True)

                logger.info("  building pipeline ...")
                bldg_config = build_config(row, eplus_dir, energy_weight)

                logger.info("  creating environment ...")
                env = create_simulator(bldg_config)
                env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

                try:
                    if hasattr(policy, "bind_env") and callable(
                        getattr(policy, "bind_env")
                    ):
                        policy.bind_env(env)

                    logger.info("  running rollout (max_steps=%d) ...", max_steps)
                    results, _data = run_rollout(
                        env=env,
                        policy=policy,
                        n_episodes=1,
                        deterministic=True,
                        max_steps=max_steps,
                        record=False,
                    )
                    episode = (
                        results[0]
                        if results
                        else EpisodeResult(total_reward=0.0, n_steps=0)
                    )
                    logger.info(
                        "  done: reward=%.2f  steps=%d",
                        episode.total_reward,
                        episode.n_steps,
                    )
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

                rec = RolloutRecord(
                    building_id=bid,
                    building_type=btype,
                    place=place,
                    episode_result=episode,
                    success=True,
                    error=None,
                )

            except Exception as e:
                logger.error("  FAILED: %s", e)
                try:
                    errors_dir.mkdir(parents=True, exist_ok=True)
                    (errors_dir / f"{btype}_{bid}.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                except Exception:
                    pass

                rec = RolloutRecord(
                    building_id=bid,
                    building_type=btype,
                    place=place,
                    episode_result=None,
                    success=False,
                    error=_short_error(e),
                )

            records.append(rec)
            payload = asdict(rec)
            if rec.episode_result is not None:
                payload["episode_result"] = asdict(rec.episode_result)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

    _log_summary(records)
    return records


def _log_summary(records: list[RolloutRecord]) -> None:
    n_ok = sum(1 for r in records if r.success)
    n_fail = len(records) - n_ok
    logger.info("=" * 60)
    logger.info("SUMMARY: %d OK, %d FAILED out of %d", n_ok, n_fail, len(records))

    if n_ok > 0:
        rewards = [
            r.episode_result.total_reward
            for r in records
            if r.success and r.episode_result is not None
        ]
        logger.info(
            "  reward: mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
            np.mean(rewards),
            np.std(rewards),
            np.min(rewards),
            np.max(rewards),
        )

    by_type: dict[str, list[RolloutRecord]] = {}
    for r in records:
        by_type.setdefault(r.building_type, []).append(r)

    for btype in sorted(by_type):
        type_records = by_type[btype]
        ok = sum(1 for r in type_records if r.success)
        fail = len(type_records) - ok
        type_rewards = [
            r.episode_result.total_reward
            for r in type_records
            if r.success and r.episode_result is not None
        ]
        avg = np.mean(type_rewards) if type_rewards else float("nan")
        logger.info("  %s: %d OK, %d FAIL, avg_reward=%.2f", btype, ok, fail, avg)

    if n_fail > 0:
        logger.info("-" * 60)
        logger.info("Failures:")
        for r in records:
            if not r.success:
                logger.info(
                    "  id=%-6d type=%-20s error=%s",
                    r.building_id,
                    r.building_type,
                    r.error,
                )
