#!/usr/bin/env python3
"""Validate UnitaryG36 baseline SAT control on Warehouse, RetailStandalone, RestaurantFastFood.

For each building type, loads one building from the test split, runs the
G36 piecewise-linear controller for a winter period, and prints per-zone
temperature statistics relative to the heating/cooling setpoints.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from b2b.api import make_multizones_env
from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
from b2b.benchmark.runner import run_rollout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class ZoneTemperatureStats:
    zone_name: str
    mean: float
    std: float
    min: float
    max: float
    pct_below_heating: float
    pct_above_cooling: float
    pct_in_deadband: float


def analyse_zone_temperatures(
    obs: np.ndarray,
    obs_names: list[str],
    controlled_zones: list[str],
    heating_sp: float,
    cooling_sp: float,
) -> list[ZoneTemperatureStats]:
    prefix = "zone air temperature"
    stats: list[ZoneTemperatureStats] = []
    for zone in controlled_zones:
        zn = zone.strip().lower()
        col_idx: int | None = None
        for i, name in enumerate(obs_names):
            nl = name.strip().lower()
            if nl.startswith(prefix):
                zone_part = nl[len(prefix):].strip()
                if zone_part == zn or zn in zone_part or zone_part in zn:
                    col_idx = i
                    break
        if col_idx is None:
            continue

        temps = obs[:, col_idx]
        n = len(temps)
        if n == 0:
            continue
        stats.append(ZoneTemperatureStats(
            zone_name=zone,
            mean=float(np.mean(temps)),
            std=float(np.std(temps)),
            min=float(np.min(temps)),
            max=float(np.max(temps)),
            pct_below_heating=float(np.sum(temps < heating_sp) / n * 100),
            pct_above_cooling=float(np.sum(temps > cooling_sp) / n * 100),
            pct_in_deadband=float(
                np.sum((temps >= heating_sp) & (temps <= cooling_sp)) / n * 100
            ),
        ))
    return stats


def run_one_building(
    building_type: str,
    split_index: int,
    run_period: str,
    output_dir: Path,
    heating_sp: float,
    cooling_sp: float,
) -> None:
    log.info("=" * 70)
    log.info("Building type: %s  split=test  index=%d  period=%s", building_type, split_index, run_period)
    log.info("=" * 70)

    eplus_dir = output_dir / building_type
    eplus_dir.mkdir(parents=True, exist_ok=True)

    env = make_multizones_env(
        building_type=building_type,
        split="test",
        split_index=split_index,
        eplus_output_dir=str(eplus_dir),
        task={"run_period": run_period},
    )

    try:
        meta = env.metadata
        obs_names: list[str] = meta["observation_names"]
        act_names: list[str] = meta["action_names"]
        controlled_zones: list[str] = meta.get("controlled_zones", [])
        equipment = meta.get("hvac_equipment", [])
        source = meta.get("building_source_metadata", {})

        log.info("  building_id=%s  place=%s", source.get("building_id"), source.get("place"))
        log.info("  obs_dim=%d  act_dim=%d  controlled_zones=%d",
                 len(obs_names), len(act_names), len(controlled_zones))
        log.info("  zones: %s", controlled_zones)

        unitary_count = sum(
            1 for e in equipment
            if hasattr(e, "equipment_type") and e.equipment_type == "unitarysystem"
        )
        log.info("  unitary systems: %d", unitary_count)

        from omegaconf import OmegaConf
        policy_cfg = OmegaConf.create({
            "heating_setpoint_c": heating_sp,
            "cooling_setpoint_c": cooling_sp,
            "kp": 1.0,
            "ki": 0.05,
            "min_fan_fraction": 0.15,
            "med_fan_fraction": 0.50,
            "sat_min_c": 13.0,
        })
        policy = UnitaryG36Policy(policy_cfg)

        max_steps = env.spec.max_episode_steps if env.spec and env.spec.max_episode_steps else 90 * 24 * 4

        log.info("  Running rollout for %d steps ...", max_steps)
        results, data = run_rollout(
            env=env,
            policy=policy,
            n_episodes=1,
            deterministic=True,
            max_steps=max_steps,
            record=True,
        )
        assert data is not None

        log.info("  Episode return: %.2f  steps: %d", results[0].total_reward, results[0].n_steps)

        zone_stats = analyse_zone_temperatures(
            data.obs, obs_names, controlled_zones, heating_sp, cooling_sp,
        )

        log.info("")
        log.info("  %-35s %6s %6s %6s %6s  %8s %8s %8s",
                 "Zone", "Mean", "Std", "Min", "Max", "<HtgSP", ">ClgSP", "InBand")
        log.info("  " + "-" * 110)
        for zs in zone_stats:
            log.info("  %-35s %6.1f %6.1f %6.1f %6.1f  %7.1f%% %7.1f%% %7.1f%%",
                     zs.zone_name[:35], zs.mean, zs.std, zs.min, zs.max,
                     zs.pct_below_heating, zs.pct_above_cooling, zs.pct_in_deadband)

        if zone_stats:
            avg_in_band = np.mean([zs.pct_in_deadband for zs in zone_stats])
            avg_below = np.mean([zs.pct_below_heating for zs in zone_stats])
            avg_above = np.mean([zs.pct_above_cooling for zs in zone_stats])
            log.info("  " + "-" * 110)
            log.info("  %-35s %6s %6s %6s %6s  %7.1f%% %7.1f%% %7.1f%%",
                     "AVERAGE", "", "", "", "", avg_below, avg_above, avg_in_band)
        log.info("")

    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-period", default="winter", choices=["winter", "summer", "full_year"])
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--heating-sp", type=float, default=21.0)
    parser.add_argument("--cooling-sp", type=float, default=24.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--types", nargs="+", default=["Warehouse", "RetailStandalone", "RestaurantFastFood"])
    args = parser.parse_args()

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="validate_unitary_"))
    log.info("Output directory: %s", output_dir)

    for btype in args.types:
        try:
            run_one_building(
                building_type=btype,
                split_index=args.split_index,
                run_period=args.run_period,
                output_dir=output_dir,
                heating_sp=args.heating_sp,
                cooling_sp=args.cooling_sp,
            )
        except Exception:
            log.exception("FAILED for %s", btype)


if __name__ == "__main__":
    main()
