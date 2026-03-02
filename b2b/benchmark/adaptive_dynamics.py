from __future__ import annotations

import json
import logging
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collections.abc import Callable

import gymnasium as gym

from b2b.benchmark.runner import EpisodeResult, PolicyLike, run_rollout
from b2b.make_env import make_env
from b2b.sources.single_zone_houses import SingleZoneHouseRowIdSplits
from b2b.types import RunPeriodConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdaptiveDynamicsRecord:
    split_index: int
    dataset_row_index: int
    building_id: int
    episode_result: EpisodeResult | None
    success: bool
    error: str | None
    building_source_metadata: dict[str, Any]


def _coerce_to_plain_dict(cfg: object) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    # Hydra/OmegaConf
    try:
        from omegaconf import OmegaConf  # type: ignore

        obj = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(obj, dict):
            return dict(obj)
    except Exception:
        pass
    return {}


def _short_error(e: BaseException) -> str:
    msg = str(e).strip().replace("\n", " ")
    if len(msg) > 240:
        msg = msg[:240] + "…"
    return f"{type(e).__name__}: {msg}"


def _with_train_selection(cfg: dict[str, Any], *, split_index: int) -> dict[str, Any]:
    """
    Return a copy of cfg with `bldg.selection` set so `b2b.make_env.make_env()`
    picks a deterministic HydroQuebec building from stored split lists.
    """
    return _with_selection(cfg, split="train", split_index=split_index)


def _with_selection(
    cfg: dict[str, Any], *, split: str, split_index: int
) -> dict[str, Any]:
    split_s = str(split).strip().lower()
    if split_s not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    out = deepcopy(cfg)
    bldg = out.setdefault("bldg", {})
    if not isinstance(bldg, dict):
        bldg = {}
        out["bldg"] = bldg
    bldg["split"] = split_s
    bldg["index"] = int(split_index)
    out["bldg"] = bldg
    return out


def benchmark_adaptive_dynamics(
    config: object,
    policy: PolicyLike,
    output_dir: Path,
    env_wrapper: Callable[[gym.Env], gym.Env] | None = None,
) -> list[AdaptiveDynamicsRecord]:
    """
    Sequentially evaluate a policy on the Hydro-Québec train selection list.

    For each dataset row index stored in:
      `b2b/sources/data/action_space_2_zone_1_train_data.json`
    we build the corresponding EnergyPlus environment, run a full episode
    (intended to be one year), and collect the resulting EpisodeResult.

    Results are appended to `<output_dir>/adaptive_dynamics_results.jsonl`.

    If *env_wrapper* is provided it is called on every freshly-created
    environment **before** the rollout, e.g. to apply observation padding,
    augmentation, or normalisation so that a trained RL model sees the same
    observation space it was trained on.
    """
    cfg = _coerce_to_plain_dict(config)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_section = cfg.get("benchmark", {})
    if not isinstance(bench_section, dict):
        bench_section = {}
    split = str(bench_section.get("split", "train")).strip().lower()
    if split not in ("train", "test"):
        raise ValueError(f"benchmark.split must be 'train' or 'test', got {split!r}")
    start = int(bench_section.get("start", 0) or 0)
    limit_raw = bench_section.get("limit", 0) or 0
    try:
        limit = int(limit_raw)
    except Exception:
        limit = 0
    max_steps_raw = bench_section.get("max_steps", None)
    try:
        max_steps = int(max_steps_raw) if max_steps_raw is not None else None
    except Exception:
        max_steps = None

    if start < 0:
        start = 0

    splits = SingleZoneHouseRowIdSplits.load_from_action_space_2_zone_1()
    row_indices = (
        list(map(int, splits.train_row_ids))
        if split == "train"
        else list(map(int, splits.test_row_ids))
    )

    end = len(row_indices) if limit <= 0 else min(len(row_indices), start + limit)

    results_path = output_dir / "adaptive_dynamics_results.jsonl"
    records: list[AdaptiveDynamicsRecord] = []

    # Each element of train_row_ids is a dataset row index (0-based). The split index
    # is what `make_env` expects via bldg.split / bldg.index.
    for split_index in range(start, end):
        dataset_row_index = int(row_indices[int(split_index)])
        building_id = int(dataset_row_index) + 1
        eplus_output_dir = output_dir / "eplus_outputs"

        episode_result: EpisodeResult | None = None
        building_source_metadata: dict[str, Any] = {
            "source": "hydroquebec",
            "dataset_row_index": int(dataset_row_index),
            "building_id": int(building_id),
            "split": str(split),
            "split_index": int(split_index),
        }

        try:
            env_cfg = _with_selection(cfg, split=split, split_index=split_index)
            env = make_env(config=env_cfg, eplus_output_dir=str(eplus_output_dir))
            if env_wrapper is not None:
                env = env_wrapper(env)
            try:
                # Prefer env-selected metadata if present.
                meta = getattr(env, "metadata", {}) or {}
                if isinstance(meta, dict):
                    src = meta.get("building_source_metadata")
                    if isinstance(src, dict):
                        building_source_metadata = dict(src)
                        # ensure split/split_index are present for traceability
                        building_source_metadata.setdefault("split", str(split))
                        building_source_metadata.setdefault(
                            "split_index", int(split_index)
                        )
                # Optional hook so policies can bind per-env metadata (action_names, etc.).
                if hasattr(policy, "bind_env") and callable(
                    getattr(policy, "bind_env")
                ):
                    try:
                        policy.bind_env(env)  # type: ignore[no-untyped-call]
                    except Exception:
                        pass
                fallback = (
                    env.spec.max_episode_steps
                    if env.spec and env.spec.max_episode_steps
                    else RunPeriodConfig.from_name("full_year").expected_steps()
                )
                cap = (
                    int(max_steps)
                    if isinstance(max_steps, int) and max_steps > 0
                    else int(fallback)
                )
                results, _data = run_rollout(
                    env=env,
                    policy=policy,
                    n_episodes=1,
                    deterministic=True,
                    max_steps=cap,
                    record=False,
                )
                episode_result = (
                    results[0]
                    if results
                    else EpisodeResult(total_reward=0.0, n_steps=0)
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            rec = AdaptiveDynamicsRecord(
                split_index=int(split_index),
                dataset_row_index=int(dataset_row_index),
                building_id=int(building_id),
                episode_result=episode_result,
                success=True,
                error=None,
                building_source_metadata=building_source_metadata,
            )
        except Exception as e:
            # Keep going; record failure for traceability.
            logger.warning(
                "Adaptive dynamics benchmark failed for split_index=%s row=%s: %s",
                split_index,
                dataset_row_index,
                e,
            )
            try:
                (output_dir / "errors").mkdir(parents=True, exist_ok=True)
                (output_dir / "errors" / f"{split}_{split_index:05d}.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            except Exception:
                pass

            rec = AdaptiveDynamicsRecord(
                split_index=int(split_index),
                dataset_row_index=int(dataset_row_index),
                building_id=int(building_id),
                episode_result=None,
                success=False,
                error=_short_error(e),
                building_source_metadata=building_source_metadata,
            )

        records.append(rec)
        # Append a JSONL record so long runs keep partial results.
        payload = asdict(rec)
        if rec.episode_result is not None:
            payload["episode_result"] = asdict(rec.episode_result)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    return records
