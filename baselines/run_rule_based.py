#!/usr/bin/env python3
"""Evaluate rule-based controllers and generate baseline_returns.csv.

Usage with Hydra::

    python -m baselines.run_rule_based experiment=eval_rule_based
    python -m baselines.run_rule_based experiment=eval_rule_based \
        building_types=[OfficeSmall] tasks=[task1] max_buildings_per_type=5
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

import building2building as b2b
from baselines.controllers.ashrae_air_loop import (
    AshraeAirLoopConfig,
    AshraeAirLoopPolicy,
)
from baselines.controllers.unitary_g36 import UnitaryG36Config, UnitaryG36Policy
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


def _load_tuned_g36(bt: str, cz: int | None) -> UnitaryG36Config:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"unitary_g36_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = yaml.safe_load(p.read_text())
            raw.pop("type", None)
            return UnitaryG36Config(**{
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in raw.items()
                if k != "target_schedule"
            })
    return UnitaryG36Config()


def _load_tuned_air_loop(bt: str, cz: int | None) -> AshraeAirLoopConfig:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"ashrae_air_loop_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = yaml.safe_load(p.read_text())
            raw.pop("type", None)
            return AshraeAirLoopConfig(**raw)
    return AshraeAirLoopConfig()


def _get_climate_zone(bt: str, bid: str) -> int | None:
    try:
        from building2building.sources.multizones_reference_buildings import (
            climate_zone_for_building,
        )
        return climate_zone_for_building(bt, int(bid))
    except Exception:
        return None


def _select_policy(
    building_type: str, building_id: str, env: Any
) -> UnitaryG36Policy | AshraeAirLoopPolicy:
    cz = _get_climate_zone(building_type, building_id)
    if building_type in VAV_BUILDING_TYPES:
        policy = AshraeAirLoopPolicy(_load_tuned_air_loop(building_type, cz))
    else:
        policy = UnitaryG36Policy(_load_tuned_g36(building_type, cz))
    policy.bind_env(env)
    return policy


def evaluate_building(
    building_type: str,
    building_id: str,
    task: str,
    *,
    n_runs: int = 1,
) -> RunResult:
    """Run the rule-based controller on one building and return results."""
    rewards: list[float] = []

    for run_idx in range(n_runs):
        env = b2b.new_make_env(
            building_type,
            building_id=building_id,
            task=task,
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
    max_bldgs = cfg.get("max_buildings_per_type")
    n_runs: int = int(cfg.get("n_runs", 1))
    output_csv = Path(str(cfg.get("output_csv", "baseline_returns.csv")))

    results: list[RunResult] = []

    for bt in building_types:
        building_ids = b2b.list_buildings(bt, split=split)
        if max_bldgs is not None:
            building_ids = building_ids[: int(max_bldgs)]

        logger.info(
            "Evaluating %s: %d buildings x %d tasks",
            bt, len(building_ids), len(tasks),
        )

        for bid in building_ids:
            for task in tasks:
                try:
                    result = evaluate_building(bt, bid, task, n_runs=n_runs)
                    results.append(result)
                except Exception:
                    logger.exception("Failed: %s/%s task=%s", bt, bid, task)

    if results:
        write_csv(results, output_csv, n_runs=n_runs)
    else:
        logger.warning("No results to write.")


if __name__ == "__main__":
    main()
