"""Produce ``building2building/data/reward_normalizers.yaml``.

Rolls out the reference reactive controllers (the same ones behind
``baseline_returns.csv`` -- see :mod:`baselines.run_reactive_control`)
on the calibration task (``task_occ_e0``: occupancy regime, seasonal
unoccupied policy), records the per-building mean
``(temp_penalty, power_penalty)``, and writes the median per
``(building_type, climate_zone)`` bucket into one seasonal YAML section
per run period.

The reward has an asymmetric normalization (see
:mod:`building2building.data.reward_normalizers`):

* **Comfort is unnormalized:** ``tau_T`` is pinned to ``1.0`` for every
  bucket, so ``temp_penalty`` is a raw squared out-of-band deviation in
  degC^2 -- the same physical unit everywhere.
* **Energy is normalized:** ``tau_E`` is the reference controller's
  energy spend, so ``power_penalty / tau_E = 1`` means "spends like the
  reference controller for this bucket".

By default this writes the packaged
``building2building/data/reward_normalizers.yaml`` in place.

Usage::

    # regenerate the packaged normalizers for all validated periods
    python -m baselines.compute_reactive_reward_normalizers \\
        --run-periods winter summer full_year --max-per-type 64 \\
        --n-workers 8 --mode all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from building2building.config.tasks import TaskPreset, resolve_task_preset
from building2building.types import NormalizedDeadbandRewardConfig

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

GENERATOR_MODULE = "baselines.compute_reactive_reward_normalizers"

# Placeholder normalizers for env construction only: the reactive
# controller never reads the scalar reward, and (temp_penalty,
# power_penalty) are recomputed from raw_observation.
_CALIBRATION_TAU_PLACEHOLDER = 1.0

DEFAULT_OUTPUT_YAML = (
    Path(__file__).resolve().parents[1]
    / "building2building"
    / "data"
    / "reward_normalizers.yaml"
)
DEFAULT_PLOT_PATH = (
    Path("experiences")
    / "figures"
    / "reward"
    / "fig_reactive_policy_normalizer_calibration.png"
)


def _default_data_dir() -> Path:
    scratch = os.environ.get("SCRATCH")
    root = Path(scratch) if scratch else Path("/tmp")
    return root / "b2b" / "reward_normalizers_reactive" / "data"


@lru_cache(maxsize=1)
def _calibration_task_preset() -> TaskPreset:
    """Calibration preset with tau_T=tau_E=1 so env build needs no YAML."""
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

    import building2building as b2b
    from baselines.run_reactive_control import _select_policy
    from baselines.utils.evaluation import run_episode

    try:
        env = b2b.make_env(
            spec.building_type,
            building_id=spec.building_id,
            task=_calibration_task_preset(),
            run_period=spec.run_period,  # type: ignore[arg-type]
        )
    except Exception as exc:
        return f"fail: env_init: {exc}"

    try:
        policy = _select_policy(spec.building_type, spec.building_id, env)
        result = run_episode(env, policy)
        mean_t, mean_e, n_steps = mean_deadband_penalties_from_infos(
            result.infos, env.unwrapped.reward_fn
        )

        stats = BuildingStats(
            run_period=spec.run_period,
            building_type=spec.building_type,
            building_id=spec.building_id,
            cz_key=cz_key_for(spec.building_type, spec.building_id),
            mean_temp_penalty=mean_t,
            mean_power_penalty=mean_e,
            n_steps=n_steps,
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
            "Compute (tau_T, tau_E) from tuned-reactive-controller rollouts "
            f"on {CALIBRATION_TASK}."
        )
    )
    p.add_argument("--mode", choices=("all", "rollout", "aggregate"), default="all")
    p.add_argument(
        "--run-periods",
        nargs="+",
        choices=list(DEFAULT_RUN_PERIODS),
        # Default to every period the packaged YAML ships: --output-yaml
        # points at the packaged file, and writing a subset of periods
        # would drop the missing sections and break env builds for them.
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

    data_dir = args.data_dir if args.data_dir is not None else _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Per-building cache dir: %s", data_dir)

    run_periods = list(args.run_periods)

    if args.mode in ("all", "rollout"):
        picks = list_train_buildings(
            args.building_types,
            args.max_per_type,
            climate_zone=args.climate_zone,
        )
        shard = shard_picks(picks, args.shard_index, args.shard_count)
        specs = build_specs(shard, data_dir, run_periods)
        logger.info(
            "Planned %d buildings, %d run periods, %d specs this worker.",
            len(shard),
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
            logger.warning("%d rollouts failed (of %d).", len(failed), len(statuses))
            for k in failed[:5]:
                logger.warning("  %s -> %s", k, statuses[k])

    if args.mode in ("all", "aggregate"):
        stats = load_stats(data_dir, run_periods)
        logger.info("Loaded %d per-building stats from %s", len(stats), data_dir)
        if not stats:
            logger.error("No stats under %s; run rollout shards first.", data_dir)
            return 1

        aggregated = aggregate(stats)
        # Comfort is unnormalized: pin tau_T=1 for every bucket so
        # temp_penalty stays in raw degC^2. Only tau_E (the reference
        # controller's energy spend) is calibrated.
        for period_payload in aggregated.values():
            for payload in period_payload.values():
                payload["tau_T"] = 1.0
                payload["tau_T_iqr"] = 0.0
        save_calibration_plot(
            aggregated,
            args.plot_path,
            run_periods,
            title="Reference reactive-controller energy-normalizer calibration",
        )
        clean = strip_internal_keys(aggregated)
        write_reward_normalizers_yaml(
            clean,
            args.output_yaml,
            run_periods,
            generator_module=GENERATOR_MODULE,
            source_lines=[
                "controller: reactive",
                "policy: baselines.run_reactive_control reference RBCs (raw action "
                "space); tau_T pinned to 1.0 (unnormalized comfort)",
                f"calibration_task: {CALIBRATION_TASK}",
                f"run_periods: [{', '.join(run_periods)}]",
                f"split: {CALIBRATION_SPLIT}",
                "aggregation: median_over_buildings",
            ],
        )
        logger.info("Wrote %s", args.output_yaml)

    return 0


if __name__ == "__main__":
    sys.exit(main())
