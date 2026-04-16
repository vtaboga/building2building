#!/usr/bin/env python3
"""Evaluate rule-based controllers and generate baseline_returns.csv.

Usage with Hydra::

    python -m baselines.run_rule_based experiment=eval_rule_based
    python -m baselines.run_rule_based experiment=eval_rule_based \
        building_types=[OfficeSmall] tasks=[task1] max_buildings_per_type=5

Save trajectories and plot temperature / actuator time-series::

    python -m baselines.run_rule_based experiment=eval_rule_based \
        save_trajectories=true plot_trajectories=true \
        building_types=[OfficeSmall] tasks=[task1] max_buildings_per_type=1
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
    rewards: list[float]
    reward_mean: float


def _load_tuned_unitary_hvac(bt: str, cz: int | None) -> UnitaryHvacConfig:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"unitary_hvac_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = yaml.safe_load(p.read_text())
            raw.pop("type", None)
            return UnitaryHvacConfig(**{
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in raw.items()
                if k != "target_schedule"
            })
    return UnitaryHvacConfig()


def _load_tuned_air_loop(bt: str, cz: int | None) -> AirLoopConfig:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"air_loop_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = yaml.safe_load(p.read_text())
            raw.pop("type", None)
            return AirLoopConfig(**raw)
    return AirLoopConfig()


PLACE_TO_CLIMATE_ZONE: dict[str, int] = {
    "Miami": 1,
    "Houston": 2,
    "Tampa": 2,
    "Tucson": 2,
    "Atlanta": 3,
    "ElPaso": 3,
    "SanDiego": 3,
    "SanFrancisco": 3,
    "Albuquerque": 4,
    "Baltimore": 4,
    "NewYork": 4,
    "PortAngeles": 4,
    "Seattle": 4,
    "Buffalo": 5,
    "Chicago": 5,
    "Denver": 5,
    "Vancouver": 5,
    "GreatFalls": 6,
    "Rochester": 6,
    "Duluth": 7,
    "InternationalFalls": 7,
    "Fairbanks": 8,
}


def _get_climate_zone(bt: str, bid: str) -> int | None:
    """Best-effort climate zone lookup from the building's weather filename.

    Returns ``None`` when no place-to-CZ mapping is available (e.g.
    for SingleFamilyHouse buildings that use per-building weather).
    """
    try:
        from building2building.data.registry import get_registry

        info = get_registry().get_building_by_id(bt, bid)
        weather = info.weather_file
        if not weather:
            return None
        place = Path(weather).stem.split("_")[0]
        return PLACE_TO_CLIMATE_ZONE.get(place)
    except Exception:
        return None


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
    """Run the rule-based controller on one building and return results.

    When *save_trajectories* is ``True``, each episode's full observation /
    action / reward arrays are written as ``.npz`` files under *trajectory_dir*.
    When *plot_trajectories* is ``True``, temperature and actuator time-series
    figures are saved under *plot_dir*.
    """
    rewards: list[float] = []

    for run_idx in range(n_runs):
        env = b2b.new_make_env(
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
        rewards=rewards,
        reward_mean=float(np.mean(rewards)),
    )


def write_csv(results: list[RunResult], path: Path, *, n_runs: int) -> None:
    """Write results to CSV with per-task rows."""
    run_cols = [f"reward_run{i + 1}" for i in range(n_runs)]
    fieldnames = ["building_type", "task", "building_id"] + run_cols + [
        "reward_mean"
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.building_type, x.task, x.building_id)):
            row: dict[str, Any] = {
                "building_type": r.building_type,
                "task": r.task,
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
    run_period_raw = str(cfg.get("run_period", "full_year"))
    allowed_run_periods = {"full_year", "winter", "summer"}
    if run_period_raw not in allowed_run_periods:
        raise ValueError(
            f"Invalid run_period '{run_period_raw}'. "
            f"Expected one of {sorted(allowed_run_periods)}."
        )
    run_period: Literal["full_year", "winter", "summer"] = run_period_raw  # type: ignore[assignment]
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
            "Evaluating %s: %d buildings x %d tasks (run_period=%s)",
            bt,
            len(building_ids),
            len(tasks),
            run_period,
        )

        for bid in building_ids:
            for task in tasks:
                try:
                    result = evaluate_building(
                        bt,
                        bid,
                        task,
                        run_period=run_period,
                        n_runs=n_runs,
                        save_trajectories=save_trajectories,
                        plot_trajectories=plot_trajectories,
                        trajectory_dir=trajectory_dir,
                        plot_dir=plot_dir,
                    )
                    results.append(result)
                except Exception:
                    logger.exception("Failed: %s/%s task=%s", bt, bid, task)

    if results:
        write_csv(results, output_csv, n_runs=n_runs)
    else:
        logger.warning("No results to write.")


if __name__ == "__main__":
    main()
