"""Compute reward normalizers from SAC warmup uniform-random rollouts.

Rolls out ``env.action_space.sample()`` on the RL wrapper stack
(``rescale_action=True``, ``normalize_obs=True``) under ``task_occ_emed``,
recomputes ``(temp_penalty, power_penalty)`` from ``raw_observation``, and
writes ``building2building/data/reward_normalizers.yaml``.

Usage::

    python -m baselines.compute_random_policy_reward_normalizers --mode rollout \\
        --shard-index 1 --shard-count 48 --n-workers 8

    python -m baselines.compute_random_policy_reward_normalizers --mode aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import traceback
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from baselines.utils.reward_normalizer_calibration import (
    CALIBRATION_SPLIT,
    CALIBRATION_TASK,
    DEFAULT_BUILDING_TYPES,
    DEFAULT_RUN_PERIODS,
    BuildingStats,
    RolloutSpec,
    aggregate,
    build_specs,
    cz_key_for,
    default_data_dir,
    list_train_buildings,
    load_stats,
    mean_deadband_penalties_from_infos,
    run_rollouts,
    save_calibration_plot,
    shard_picks,
    strip_internal_keys,
    write_reward_normalizers_yaml,
)

logger = logging.getLogger(__name__)

GENERATOR_MODULE = "baselines.compute_random_policy_reward_normalizers"

# Placeholder normalizers for env construction only.  Rollouts recompute
# (temp_penalty, power_penalty) from raw_observation; the scalar reward
# from env.step is never read.
_CALIBRATION_TAU_PLACEHOLDER = 1.0

DEFAULT_OUTPUT_YAML = (
    Path("building2building") / "data" / "reward_normalizers.yaml"
)
DEFAULT_PLOT_PATH = (
    Path("analysis") / "task_study" / "reward_design" / "plots"
    / "fig_random_policy_normalizer_calibration.png"
)


def _seed_for(building_type: str, building_id: str) -> int:
    key = f"{building_type}/{building_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


@lru_cache(maxsize=1)
def _calibration_task_preset() -> TaskPreset:
    """``task_occ_emed`` with tau_T=tau_E=1 so env build needs no YAML."""
    preset = resolve_task_preset(CALIBRATION_TASK)
    reward = preset.reward
    if isinstance(reward, NormalizedDeadbandRewardConfig) and not reward.is_filled:
        preset = replace(
            preset,
            reward=reward.filled(
                _CALIBRATION_TAU_PLACEHOLDER, _CALIBRATION_TAU_PLACEHOLDER
            ),
        )
    return preset


def _run_single_rollout(spec: RolloutSpec) -> str:
    if spec.out_path.exists():
        return "skip"

    from baselines.utils.training import make_rl_env_fn

    seed = _seed_for(spec.building_type, spec.building_id)

    try:
        env = make_rl_env_fn(
            building_type=spec.building_type,
            building_id=spec.building_id,
            task=_calibration_task_preset(),
            run_period=spec.run_period,
            normalize_obs=True,
            rescale_action=True,
            monitor=False,
        )()
    except Exception as exc:
        return f"fail: env_init: {exc}"

    rf = env.unwrapped.reward_fn
    infos: list[dict] = []

    try:
        env.action_space.seed(seed)
        try:
            _obs, info = env.reset(seed=seed)
        except TypeError:
            _obs, info = env.reset()
        infos.append(info)

        done = False
        while not done:
            action = env.action_space.sample()
            _obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            infos.append(info)

        mean_t, mean_e, n_steps = mean_deadband_penalties_from_infos(infos[1:], rf)

        stats = BuildingStats(
            run_period=spec.run_period,
            building_type=spec.building_type,
            building_id=spec.building_id,
            cz_key=cz_key_for(spec.building_type, spec.building_id),
            mean_temp_penalty=mean_t,
            mean_power_penalty=mean_e,
            n_steps=n_steps,
            seed=seed,
        )
        spec.out_path.parent.mkdir(parents=True, exist_ok=True)
        spec.out_path.write_text(json.dumps(stats.to_json(), indent=2))
        return "ok"
    except Exception as exc:
        tb = traceback.format_exc(limit=3).strip().replace("\n", " | ")
        return f"fail: rollout: {exc} | {tb}"
    finally:
        try:
            env.close()
        except Exception:
            pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute (tau_T, tau_E) from SAC-warmup uniform-random rollouts "
            f"on {CALIBRATION_TASK}."
        )
    )
    p.add_argument(
        "--mode",
        choices=("all", "rollout", "aggregate"),
        default="all",
    )
    p.add_argument(
        "--run-periods",
        nargs="+",
        choices=list(DEFAULT_RUN_PERIODS),
        default=list(DEFAULT_RUN_PERIODS),
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT_YAML)
    p.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    p.add_argument(
        "--building-types",
        nargs="+",
        default=list(DEFAULT_BUILDING_TYPES),
    )
    p.add_argument("--climate-zone", type=int, default=None)
    p.add_argument("--max-per-type", type=int, default=None)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=1)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    if args.data_dir is None and "SCRATCH" not in os.environ:
        logger.warning(
            "$SCRATCH is not set; using %s for per-building cache.",
            data_dir,
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Per-building cache dir: %s", data_dir)

    run_periods = list(args.run_periods)

    if args.mode in ("all", "rollout"):
        picks = list_train_buildings(
            args.building_types,
            args.max_per_type,
            climate_zone=args.climate_zone,
        )
        shard_picks_list = shard_picks(picks, args.shard_index, args.shard_count)
        specs = build_specs(shard_picks_list, data_dir, run_periods)
        logger.info(
            "Planned %d train buildings, %d run periods, %d specs this worker.",
            len(shard_picks_list),
            len(run_periods),
            len(specs),
        )
        if args.dry_run:
            for spec in specs:
                print(spec.key, "->", spec.out_path)
            return 0

        statuses = run_rollouts(specs, args.n_workers, _run_single_rollout)
        failed = [k for k, v in statuses.items() if v.startswith("fail")]
        if failed:
            logger.warning(
                "%d rollouts failed (of %d).", len(failed), len(statuses)
            )

    if args.mode in ("all", "aggregate"):
        stats = load_stats(data_dir, run_periods)
        logger.info("Loaded %d per-building stats from %s", len(stats), data_dir)
        if not stats:
            logger.error(
                "No stats under %s; run rollout shards first.", data_dir
            )
            return 1

        aggregated = aggregate(stats)
        save_calibration_plot(
            aggregated,
            args.plot_path,
            run_periods,
            title="SAC-warmup random-policy reward-normalizer calibration",
        )
        logger.info("Saved sanity plot to %s", args.plot_path)

        clean = strip_internal_keys(aggregated)
        write_reward_normalizers_yaml(
            clean,
            args.output_yaml,
            run_periods,
            generator_module=GENERATOR_MODULE,
            source_lines=[
                "controller: sac_warmup_uniform_random",
                "policy: stable_baselines3_sac_learning_starts_action_space_sample",
                f"calibration_task: {CALIBRATION_TASK}",
                f"run_periods: [{', '.join(run_periods)}]",
                f"split: {CALIBRATION_SPLIT}",
                "action_space: rl_wrapped_rescaled_box",
                "seed: sha256(building_type/building_id) first 32 bits",
                "aggregation: median_over_buildings",
            ],
        )
        logger.info("Wrote %s", args.output_yaml)

    return 0


if __name__ == "__main__":
    sys.exit(main())
