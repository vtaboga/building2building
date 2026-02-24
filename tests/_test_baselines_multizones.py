#!/usr/bin/env python3
"""Quick smoke test: run 10-day baseline rollouts on the first 5 test buildings per type.

OfficeMedium uses air_loop_sat; all other types use unitary_sat.
Limits to 960 steps (10 days × 24 hours × 4 steps/hour).
"""
from __future__ import annotations

import logging
import pickle
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b2b.baselines.controllers.air_loop_sat import AirLoopSatPolicy
from b2b.baselines.controllers.unitary_sat import UnitaryAirflowFirstSatPolicy
from b2b.benchmark.rollout_multizones import build_config, select_buildings
from b2b.benchmark.runner import PolicyLike, run_rollout
from b2b.simulator import create_simulator
from b2b.sources.multizones_reference_buildings import BuildingType

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_STEPS = 10 * 24 * 4  # 10 days
N_TEST = 5
OUTPUT_DIR = Path("/tmp/test_baselines_multizones")

TYPES: list[BuildingType] = [
    "Warehouse",
    "HotelSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "b2b" / "sources" / "data"


def load_test_ids(building_type: BuildingType) -> list[int]:
    path = DATA_DIR / f"{building_type}_test_data"
    return pickle.loads(path.read_bytes())


def make_policy(building_type: BuildingType) -> PolicyLike:
    if building_type == "OfficeMedium":
        return AirLoopSatPolicy()
    return UnitaryAirflowFirstSatPolicy(_DefaultUnitaryCfg())


class _DefaultUnitaryCfg:
    """Mimics the Hydra config object with default attribute access."""

    type = "unitary_sat"
    target_temp_c = 21.0
    deadband_c = 0.5
    availability_on = 2.0
    outlet_temp_heating_c = 35.0
    outlet_temp_cooling_c = 14.0
    fan_base_kg_s = 0.3
    kp = 0.2
    ki = 0.0002
    integral_limit = 200.0
    fan_min_kg_s = None
    fan_max_kg_s = None
    sat_min_c = 10.0
    sat_max_c = 45.0
    sat_step_c = 0.2
    sat_rate_limit_c_per_step = 0.2
    sat_saturation_steps = 4
    target_schedule = None


def run_test_for_type(building_type: BuildingType) -> list[dict[str, Any]]:
    test_ids = load_test_ids(building_type)[:N_TEST]
    logger.info("Testing %s with %d buildings: %s", building_type, len(test_ids), test_ids)

    from b2b.sources.multizones_reference_buildings import search_buildings

    df = search_buildings(building_type=building_type)
    df = df[df["building_id"].isin(test_ids)].sort_values("building_id").reset_index(drop=True)

    results: list[dict[str, Any]] = []
    for _, row_series in df.iterrows():
        row = dict(row_series)
        bid = int(row["building_id"])
        logger.info("  [%s] building_id=%d  place=%s", building_type, bid, row["place"])

        eplus_dir = OUTPUT_DIR / f"{building_type}_{bid}_{uuid.uuid4().hex[:6]}"
        eplus_dir.mkdir(parents=True, exist_ok=True)

        try:
            bldg_config = build_config(row, eplus_dir)
            env = create_simulator(bldg_config)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS)

            policy = make_policy(building_type)

            try:
                if hasattr(policy, "bind_env"):
                    policy.bind_env(env)

                episodes, _data = run_rollout(
                    env=env, policy=policy, n_episodes=1,
                    deterministic=True, max_steps=MAX_STEPS, record=False,
                )
                ep = episodes[0]
                logger.info(
                    "    OK: reward=%.2f  steps=%d", ep.total_reward, ep.n_steps
                )
                results.append({
                    "building_type": building_type,
                    "building_id": bid,
                    "reward": ep.total_reward,
                    "steps": ep.n_steps,
                    "success": True,
                })
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error("    FAILED: %s", e)
            traceback.print_exc()
            results.append({
                "building_type": building_type,
                "building_id": bid,
                "reward": None,
                "steps": 0,
                "success": False,
                "error": str(e),
            })

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    for bt in TYPES:
        logger.info("=" * 60)
        logger.info("Building type: %s", bt)
        logger.info("=" * 60)
        results = run_test_for_type(bt)
        all_results.extend(results)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    n_ok = sum(1 for r in all_results if r["success"])
    n_fail = len(all_results) - n_ok
    logger.info("Total: %d OK, %d FAILED out of %d", n_ok, n_fail, len(all_results))

    for bt in TYPES:
        bt_results = [r for r in all_results if r["building_type"] == bt]
        ok = sum(1 for r in bt_results if r["success"])
        rewards = [r["reward"] for r in bt_results if r["success"] and r["reward"] is not None]
        avg = np.mean(rewards) if rewards else float("nan")
        logger.info(
            "  %s: %d/%d OK, avg_reward=%.2f", bt, ok, len(bt_results), avg
        )

    if n_fail > 0:
        logger.info("Failures:")
        for r in all_results:
            if not r["success"]:
                logger.info("  %s id=%d: %s", r["building_type"], r["building_id"], r.get("error", "unknown"))


if __name__ == "__main__":
    main()
