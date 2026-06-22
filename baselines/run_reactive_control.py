#!/usr/bin/env python3
"""Evaluate reactive controllers and generate baseline_returns.csv.

Each run produces one row per ``(building_type, task, run_period, building_id)``.

Usage with Hydra::

    python -m baselines.run_reactive_control experiment=eval_reactive_control
    python -m baselines.run_reactive_control experiment=eval_reactive_control \
        building_types=[OfficeSmall] tasks=[task_const_e0] max_buildings_per_type=5

Save trajectories and plot temperature / actuator time-series::

    python -m baselines.run_reactive_control experiment=eval_reactive_control \
        save_trajectories=true plot_trajectories=true \
        building_types=[OfficeSmall] tasks=[task_const_e0] max_buildings_per_type=1
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

import building2building as b2b
from baselines.controllers.air_loop import (
    AirLoopConfig,
    AirLoopPolicy,
)
from baselines.controllers.unitary_hvac import UnitaryHvacConfig, UnitaryHvacPolicy
from baselines.plotting.plot_trajectory import (
    extract_trajectory_data,
    plot_trajectory,
)
from baselines.utils.evaluation import EpisodeResult, run_episode

logger = logging.getLogger(__name__)

VAV_BUILDING_TYPES = {"OfficeMedium"}
TUNED_CONFIGS_DIR = Path(__file__).parent / "configs" / "tuned_controllers"


@dataclass
class RunResult:
    building_type: str
    building_id: str
    task: str
    run_period: str
    rewards: list[float]
    reward_mean: float


def _load_tuned_yaml(path: Path) -> dict[str, Any]:
    """Load a tuned-controller YAML, tolerating python/tuple tags.

    Several files under ``baselines/configs/tuned_controllers`` were
    dumped with plain ``yaml.dump`` and therefore contain
    ``!!python/tuple`` (e.g. ``target_schedule.weekend_days``), which
    ``yaml.safe_load`` refuses to construct. We use ``yaml.unsafe_load``
    here because these files are produced by our own tuning pipeline.
    Downstream callers drop the ``target_schedule`` entry anyway.
    """
    return yaml.unsafe_load(path.read_text())


def _load_tuned_unitary_hvac(bt: str, cz: int | None) -> UnitaryHvacConfig:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"unitary_hvac_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = _load_tuned_yaml(p)
            raw.pop("type", None)
            return UnitaryHvacConfig(
                **{
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in raw.items()
                    if k != "target_schedule"
                }
            )
    return UnitaryHvacConfig()


def _load_tuned_air_loop(bt: str, cz: int | None) -> AirLoopConfig:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"air_loop_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = _load_tuned_yaml(p)
            raw.pop("type", None)
            return AirLoopConfig(**raw)
    return AirLoopConfig()


def _get_climate_zone(bt: str, bid: str) -> int | None:
    """Return the ASHRAE climate zone, or ``None`` for types without one.

    Thin wrapper around :func:`building2building.api.get_climate_zone` that
    returns ``None`` (rather than raising) for building types in
    :data:`building2building.api.TYPES_WITHOUT_CLIMATE_ZONE` (e.g. SFH), so
    the ``cz | None`` contract of :func:`_load_tuned_unitary_hvac` /
    :func:`_load_tuned_air_loop` is preserved.
    """
    if bt in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
        return None
    return b2b.get_climate_zone(bt, bid)


def _select_policy(
    building_type: str, building_id: str, env: Any
) -> UnitaryHvacPolicy | AirLoopPolicy:
    cz = _get_climate_zone(building_type, building_id)
    if building_type in VAV_BUILDING_TYPES:
        policy = AirLoopPolicy(_load_tuned_air_loop(building_type, cz))
    else:
        policy = UnitaryHvacPolicy(_load_tuned_unitary_hvac(building_type, cz))
    policy.bind_env(env)
    return policy


def evaluate_building(
    building_type: str,
    building_id: str,
    task: str,
    *,
    run_period: Literal["full_year", "winter", "summer"] = "full_year",
    n_runs: int = 1,
    save_trajectories: bool = False,
    plot_trajectories: bool = False,
    trajectory_dir: Path | None = None,
    plot_dir: Path | None = None,
) -> RunResult:
    """Run the reactive controller on one building and return results.

    When *save_trajectories* is ``True``, each episode's full observation /
    action / reward arrays are written as ``.npz`` files under *trajectory_dir*.
    When *plot_trajectories* is ``True``, temperature and actuator time-series
    figures are saved under *plot_dir*.
    """
    rewards: list[float] = []

    for run_idx in range(n_runs):
        env = b2b.make_env(
            building_type,
            building_id=building_id,
            task=task,
            run_period=run_period,
        )
        try:
            policy = _select_policy(building_type, building_id, env)
            result: EpisodeResult = run_episode(env, policy)
            rewards.append(result.total_reward)
            logger.info(
                "  %s/%s task=%s run=%d reward=%.1f",
                building_type,
                building_id,
                task,
                run_idx,
                result.total_reward,
            )

            if save_trajectories or plot_trajectories:
                traj = extract_trajectory_data(
                    result.observations,
                    result.actions,
                    result.rewards,
                    env.metadata,
                    building_type=building_type,
                    building_id=building_id,
                    task=task,
                )
                stem = f"{building_type}_{building_id}_{task}_run{run_idx}"

                if save_trajectories and trajectory_dir is not None:
                    traj_path = trajectory_dir / f"{stem}.npz"
                    traj.save(traj_path)
                    logger.info("    Saved trajectory → %s", traj_path)

                if plot_trajectories and plot_dir is not None:
                    fig_path = plot_dir / stem
                    plot_trajectory(traj, output_path=fig_path)
                    logger.info("    Saved plot → %s.*", fig_path)
        finally:
            env.close()

    return RunResult(
        building_type=building_type,
        building_id=building_id,
        task=task,
        run_period=run_period,
        rewards=rewards,
        reward_mean=float(np.mean(rewards)),
    )


def write_csv(results: list[RunResult], path: Path, *, n_runs: int) -> None:
    """Write results to CSV with one row per (building, task, run_period)."""
    run_cols = [f"reward_run{i + 1}" for i in range(n_runs)]
    fieldnames = [
        "building_type",
        "task",
        "run_period",
        "building_id",
        *run_cols,
        "reward_mean",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(
            results,
            key=lambda x: (x.building_type, x.task, x.run_period, x.building_id),
        ):
            row: dict[str, Any] = {
                "building_type": r.building_type,
                "task": r.task,
                "run_period": r.run_period,
                "building_id": r.building_id,
                "reward_mean": f"{r.reward_mean:.1f}",
            }
            for i, rw in enumerate(r.rewards):
                row[f"reward_run{i + 1}"] = f"{rw:.1f}"
            writer.writerow(row)

    logger.info("Wrote %d rows to %s", len(results), path)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    building_types: list[str] = list(cfg.building_types)
    tasks: list[str] = list(cfg.tasks)
    split: str = cfg.get("split", "test")
    run_periods_raw = list(cfg.run_periods)
    allowed_run_periods = {"full_year", "winter", "summer"}
    bad = [p for p in run_periods_raw if p not in allowed_run_periods]
    if bad:
        raise ValueError(
            f"Invalid run_periods {bad}. "
            f"Expected subset of {sorted(allowed_run_periods)}."
        )
    run_periods: list[Literal["full_year", "winter", "summer"]] = list(
        run_periods_raw
    )  # type: ignore[assignment]
    max_bldgs = cfg.get("max_buildings_per_type")
    n_runs: int = int(cfg.get("n_runs", 1))
    output_csv = Path(str(cfg.get("output_csv", "baseline_returns.csv")))

    save_trajectories: bool = bool(cfg.get("save_trajectories", False))
    plot_trajectories: bool = bool(cfg.get("plot_trajectories", False))
    trajectory_dir = Path(str(cfg.get("trajectory_dir", "trajectories")))
    plot_dir = Path(str(cfg.get("plot_dir", "plots")))

    if save_trajectories:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Trajectories will be saved to %s", trajectory_dir)
    if plot_trajectories:
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Trajectory plots will be saved to %s", plot_dir)

    results: list[RunResult] = []

    for bt in building_types:
        building_ids = b2b.list_buildings(bt, split=split)
        if max_bldgs is not None:
            building_ids = building_ids[: int(max_bldgs)]

        logger.info(
            "Evaluating %s: %d buildings x %d tasks x %d run_periods=%s",
            bt,
            len(building_ids),
            len(tasks),
            len(run_periods),
            run_periods,
        )

        for bid in building_ids:
            for task in tasks:
                for period in run_periods:
                    try:
                        result = evaluate_building(
                            bt,
                            bid,
                            task,
                            run_period=period,
                            n_runs=n_runs,
                            save_trajectories=save_trajectories,
                            plot_trajectories=plot_trajectories,
                            trajectory_dir=trajectory_dir,
                            plot_dir=plot_dir,
                        )
                        results.append(result)
                    except Exception:
                        logger.exception(
                            "Failed: %s/%s task=%s run_period=%s",
                            bt,
                            bid,
                            task,
                            period,
                        )

    if results:
        write_csv(results, output_csv, n_runs=n_runs)
    else:
        logger.warning("No results to write.")


if __name__ == "__main__":
    main()
