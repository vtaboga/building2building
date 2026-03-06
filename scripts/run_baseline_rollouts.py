"""Run baseline rollouts for specific building IDs and produce a CSV of returns.

Each building is rolled out 3 times with a deadband reward (occupancy-based
targets, dT=21.0, occupied=21.0 C, unoccupied=18.0 C, energy_weight=0.001).

The appropriate baseline controller is selected per building type:
  - OfficeMedium  -> ashrae_air_loop
  - all others    -> unitary_g36

Usage:
    python scripts/run_baseline_rollouts.py [--output baseline_returns.csv]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from b2b.api import make_env
from b2b.baselines.controllers.ashrae_air_loop import AshraeAirLoopPolicy
from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
from b2b.benchmark.runner import run_rollout
from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig
from b2b.types import DeadbandRewardConfig, TaskConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BUILDINGS: list[tuple[str, int]] = [
    ("RetailStandalone", 2801),
    ("RetailStandalone", 2802),
    ("RetailStandalone", 2803),
    ("RetailStandalone", 2804),
    ("RetailStandalone", 2805),
    ("RestaurantFastFood", 3801),
    ("RestaurantFastFood", 3802),
    ("RestaurantFastFood", 3803),
    ("RestaurantFastFood", 3804),
    ("RestaurantFastFood", 3805),
    ("OfficeMedium", 4801),
    ("OfficeMedium", 4802),
    ("OfficeMedium", 4803),
    ("OfficeMedium", 4804),
    ("OfficeMedium", 4805),
    ("OfficeSmall", 5801),
    ("OfficeSmall", 5802),
    ("OfficeSmall", 5803),
    ("OfficeSmall", 5804),
    ("OfficeSmall", 5805),
]

N_RUNS = 3

REWARD_CONFIG = DeadbandRewardConfig(energy_weight=0.001, dT=21.0)

TASK_CONFIG = TaskConfig.from_dict(
    {
        "run_period": "full_year",
        "target_temperature_mode": "occupancy",
        "default_zone_target_temperature": {
            "occupied_c": 21.0,
            "unoccupied_c": 18.0,
        },
    }
)

MAX_STEPS = TASK_CONFIG.expected_steps()

UNITARY_G36_POLICY_CFGS: dict[str, SimpleNamespace] = {
    "RetailStandalone": SimpleNamespace(
        type="unitary_g36",
        heating_setpoint_c=20.0,
        cooling_setpoint_c=22.0,
        kp=0.6664,
        ki=0.006335,
        integral_max=395.0,
        min_fan_fraction=0.3866,
        sat_min_c=10.87,
        sat_max_c=40.36,
        sat_initial_c=26.85,
        sat_trim=0.186,
        sat_respond=2.929,
        demand_deadband=0.101,
        availability_on=2.0,
        target_schedule=SimpleNamespace(enabled=False),
    ),
    "RestaurantFastFood": SimpleNamespace(
        type="unitary_g36",
        heating_setpoint_c=20.0,
        cooling_setpoint_c=22.0,
        kp=0.3613,
        ki=0.014485,
        integral_max=351.8,
        min_fan_fraction=0.1827,
        sat_min_c=12.85,
        sat_max_c=31.9,
        sat_initial_c=25.38,
        sat_trim=0.165,
        sat_respond=2.681,
        demand_deadband=0.139,
        availability_on=2.0,
        target_schedule=SimpleNamespace(enabled=False),
    ),
    "OfficeSmall": SimpleNamespace(
        type="unitary_g36",
        heating_setpoint_c=20.0,
        cooling_setpoint_c=22.0,
        kp=0.25,
        ki=0.02,
        integral_max=200.0,
        min_fan_fraction=0.15,
        sat_min_c=12.0,
        sat_max_c=35.0,
        sat_initial_c=21.0,
        sat_trim=0.5,
        sat_respond=1.0,
        demand_deadband=0.3,
        availability_on=2.0,
        target_schedule=SimpleNamespace(enabled=False),
    ),
}

ASHRAE_AIR_LOOP_POLICY_CFG = SimpleNamespace(
    type="ashrae_air_loop",
    target_temp=21.0,
    deadband=1.0,
    sat_aware_flow=True,
    sat_neutral=20.5,
    sat_kp=0.8,
    sat_min=10.0,
    sat_max=60.0,
    sat_rate_limit=0.3,
    outdoor_sat_gain=0.0,
    sat_cold_bias=0.0,
    sat_warm_bias=0.6,
    flow_base=0.4,
    flow_kp=0.2,
    flow_ki=0.025,
    flow_min=0.1,
    flow_max=1.0,
    flow_rate_limit=0.04,
    integral_max=15.0,
    integral_decay=0.99,
    reheat_sp_min=10.0,
    reheat_sp_max=25.0,
    reheat_sp_deadband=0.3,
    reheat_sp_kp=3.0,
    reheat_sp_rate_limit=0.2,
    clg_sp_default=22.0,
    error_ema_alpha=0.3,
)


def _build_env(building_type: str, building_id: int, run_dir: Path) -> Any:
    selection = DatasetSelectionConfig(
        dataset="multizones_reference_buildings",
        building_type=building_type,  # type: ignore[arg-type]
        split=None,
        mode="building_id",
        building_id=building_id,
    )
    build_cfg = EnvBuildConfig(
        dataset_selection=selection,
        task=TASK_CONFIG,
        reward=REWARD_CONFIG,
        env_max_steps=MAX_STEPS,
    )
    eplus_dir = run_dir / "eplus_outputs" / str(uuid.uuid4())
    eplus_dir.mkdir(parents=True, exist_ok=True)
    return make_env(build_cfg, eplus_output_dir=eplus_dir)


def _build_policy(building_type: str) -> Any:
    if building_type == "OfficeMedium":
        return AshraeAirLoopPolicy(ASHRAE_AIR_LOOP_POLICY_CFG)
    cfg = UNITARY_G36_POLICY_CFGS.get(building_type)
    if cfg is None:
        raise ValueError(f"No policy config for building type: {building_type!r}")
    return UnitaryG36Policy(cfg)


def run_single(
    building_type: str,
    building_id: int,
    run_dir: Path,
) -> float:
    """Run one full-year rollout and return the episode return (sum of rewards)."""
    env = _build_env(building_type, building_id, run_dir)
    try:
        policy = _build_policy(building_type)
        results, _ = run_rollout(
            env=env,
            policy=policy,
            n_episodes=1,
            deterministic=True,
            max_steps=MAX_STEPS,
            record=False,
        )
        return results[0].total_reward
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline rollouts for specific building IDs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baseline_returns.csv"),
        help="Output CSV path (default: baseline_returns.csv)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/baseline_rollouts"),
        help="Directory for EnergyPlus outputs (default: outputs/baseline_rollouts)",
    )
    args = parser.parse_args()

    output_path: Path = args.output
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(BUILDINGS)

    for idx, (btype, bid) in enumerate(BUILDINGS, 1):
        logger.info(
            "[%d/%d] Running %s id=%d (%d runs)...", idx, total, btype, bid, N_RUNS
        )
        rewards: list[float] = []
        for run_i in range(1, N_RUNS + 1):
            logger.info("  Run %d/%d for %s id=%d", run_i, N_RUNS, btype, bid)
            ret = run_single(btype, bid, run_dir)
            rewards.append(ret)
            logger.info("  Run %d return: %.1f", run_i, ret)

        mean_reward = float(np.mean(rewards))
        row: dict[str, Any] = {
            "building_type": btype,
            "building_id": bid,
        }
        for i, r in enumerate(rewards, 1):
            row[f"reward_run{i}"] = round(r, 1)
        row["reward_mean"] = round(mean_reward, 1)
        rows.append(row)

        logger.info(
            "  %s id=%d: runs=%s mean=%.1f",
            btype,
            bid,
            [round(r, 1) for r in rewards],
            mean_reward,
        )

    fieldnames = [
        "building_type",
        "building_id",
        *[f"reward_run{i}" for i in range(1, N_RUNS + 1)],
        "reward_mean",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
