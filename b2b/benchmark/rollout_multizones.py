"""Run policy rollouts on multizones_reference_buildings.

This module is benchmark orchestration only:
- building selection for benchmark rows
- environment creation via typed public API
- rollout execution and result aggregation
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from b2b.api import make_env
from b2b.baselines import make_policy_from_config
from b2b.benchmark.runner import EpisodeResult, PolicyLike, run_rollout
from b2b.config import DatasetSelectionConfig, EnvBuildConfig
from b2b.sources.multizones_reference_buildings import (
    BuildingType,
    search_buildings,
)
from b2b.types import TaskConfig, reward_config_from_dict

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


def select_buildings(
    building_type: BuildingType,
    n: int,
    run_period: str = "full_year",
) -> list[dict[str, Any]]:
    """Return the first *n* buildings of *building_type*, sorted by building_id."""
    df = search_buildings(building_type=building_type, run_period=run_period)
    df = df.sort_values("building_id").head(n).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(dict(row))
    return rows


def run_multizones_rollout(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
) -> list[RolloutRecord]:
    """Evaluate a baseline policy on multizones_reference_buildings.

    Reads ``multizones.types``, ``multizones.n_per_type``, and
    ``env.max_steps`` from the Hydra config.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(cfg, dict):
        cfg_any = OmegaConf.to_container(cfg, resolve=True)
        cfg = cfg_any if isinstance(cfg_any, dict) else {}

    mz = cfg.get("multizones", {})
    mz_dict: dict[str, Any] = mz if isinstance(mz, dict) else {}

    raw_types = mz_dict.get("types", None)
    if raw_types is None or raw_types == []:
        types_to_eval: list[BuildingType] = list(ALL_BUILDING_TYPES)
    else:
        types_to_eval = list(raw_types)

    n_per_type = int(mz_dict.get("n_per_type", 5))

    max_steps_raw = cfg.get("env", {})
    if isinstance(max_steps_raw, dict):
        max_steps_raw = max_steps_raw.get("max_steps", None)
    else:
        max_steps_raw = None
    reward_sect = cfg.get("reward", {})
    reward_section = reward_sect if isinstance(reward_sect, dict) else {}

    task_sect = cfg.get("task", {})
    task_section = task_sect if isinstance(task_sect, dict) else {}
    task_cfg = TaskConfig.from_dict(task_section)
    max_steps = (
        int(max_steps_raw)
        if max_steps_raw is not None
        else task_cfg.run_period.expected_steps()
    )

    results_path = output_dir / "rollout_results.jsonl"
    errors_dir = output_dir / "errors"

    policy = make_policy_from_config(cfg)

    policy_cfg = cfg.get("policy", {})
    policy_type = policy_cfg.get("type", "unknown") if isinstance(policy_cfg, dict) else "unknown"
    logger.info("Policy: %s", policy_type)
    logger.info("Building types: %s", types_to_eval)
    logger.info("Buildings per type: %d", n_per_type)
    logger.info("Max steps: %d", max_steps)
    logger.info("Output: %s", output_dir)

    records: list[RolloutRecord] = []

    for btype in types_to_eval:
        logger.info("Selecting first %d %s buildings ...", n_per_type, btype)
        rows = select_buildings(btype, n_per_type, run_period=task_cfg.run_period.name)
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
                logger.info("  creating environment ...")
                task_cfg_typed = TaskConfig.from_dict(task_section)
                reward_cfg_typed = reward_config_from_dict(reward_section)
                env = make_env(
                    EnvBuildConfig(
                        dataset_selection=DatasetSelectionConfig(
                            dataset="multizones_reference_buildings",
                            mode="building_id",
                            building_id=bid,
                            building_type=btype,
                            split=None,
                        ),
                        task=task_cfg_typed,
                        reward=reward_cfg_typed,
                        env_max_steps=max_steps,
                    ),
                    eplus_output_dir=output_dir / "eplus_outputs",
                )

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
