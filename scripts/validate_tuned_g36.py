#!/usr/bin/env python3
"""Validate tuned G36 controllers on all test-split buildings.

Loads per-building-type configs from configs/policy/unitary_g36_{suffix}.yaml,
runs full-year simulations on all 5 test buildings per type (15 total), and
reports per-zone temperature statistics.

Usage:
    python scripts/validate_tuned_g36.py
    python scripts/validate_tuned_g36.py --run-period winter --n-buildings 2
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf

from b2b.api import make_multizones_env
from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
from b2b.benchmark.runner import run_rollout
from b2b.types import RunPeriodConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEATING_SP = 20.0
COOLING_SP = 22.0

_TYPE_CONFIGS: dict[str, str] = {
    "Warehouse": "configs/policy/unitary_g36_warehouse.yaml",
    "RetailStandalone": "configs/policy/unitary_g36_retail.yaml",
    "RestaurantFastFood": "configs/policy/unitary_g36_restaurant.yaml",
}


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


@dataclass
class BuildingResult:
    building_type: str
    building_id: str
    split_index: int
    place: str
    episode_return: float
    n_steps: int
    zone_stats: list[ZoneTemperatureStats]


def _zone_temp_indices(
    obs_names: list[str], controlled_zones: list[str]
) -> dict[str, int]:
    """Map controlled zone names to their obs column index."""
    prefix = "zone air temperature"
    mapping: dict[str, int] = {}
    for zone in controlled_zones:
        zn = zone.strip().lower()
        for i, name in enumerate(obs_names):
            nl = name.strip().lower()
            if nl.startswith(prefix):
                zone_part = nl[len(prefix) :].strip()
                if zone_part == zn or zn in zone_part or zone_part in zn:
                    mapping[zone] = i
                    break
    return mapping


def analyse_zone_temperatures(
    obs: np.ndarray,
    obs_names: list[str],
    controlled_zones: list[str],
    heating_sp: float = HEATING_SP,
    cooling_sp: float = COOLING_SP,
) -> list[ZoneTemperatureStats]:
    idx_map = _zone_temp_indices(obs_names, controlled_zones)
    stats: list[ZoneTemperatureStats] = []
    for zone, col_idx in idx_map.items():
        temps = obs[:, col_idx]
        n = len(temps)
        if n == 0:
            continue
        stats.append(
            ZoneTemperatureStats(
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
            )
        )
    return stats


def load_policy_config(config_path: Path) -> Any:
    """Load a YAML policy config as an OmegaConf object."""
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


def run_one_building(
    building_type: str,
    split_index: int,
    run_period: str,
    policy_cfg: Any,
    output_dir: Path,
) -> BuildingResult:
    eplus_dir = output_dir / building_type / f"idx{split_index}"
    eplus_dir.mkdir(parents=True, exist_ok=True)

    env = make_multizones_env(
        building_type=building_type,
        split="test",
        split_index=split_index,
        eplus_output_dir=str(eplus_dir),
        task={"run_period": run_period},
        reward={"reward_type": "DeadbandRewardConfig", "energy_weight": 0.01, "dT": 1.0},
    )
    try:
        meta = env.metadata
        obs_names: list[str] = meta["observation_names"]
        controlled_zones: list[str] = meta.get("controlled_zones", [])
        source = meta.get("building_source_metadata", {})

        policy = UnitaryG36Policy(policy_cfg)
        max_steps = (
            env.spec.max_episode_steps
            if env.spec and env.spec.max_episode_steps
            else RunPeriodConfig.from_name("full_year").expected_steps()
        )

        results, data = run_rollout(
            env=env,
            policy=policy,
            n_episodes=1,
            deterministic=True,
            max_steps=max_steps,
            record=True,
        )
        assert data is not None

        zone_stats = analyse_zone_temperatures(
            data.obs, obs_names, controlled_zones
        )

        return BuildingResult(
            building_type=building_type,
            building_id=str(source.get("building_id", "?")),
            split_index=split_index,
            place=str(source.get("place", "?")),
            episode_return=results[0].total_reward,
            n_steps=results[0].n_steps,
            zone_stats=zone_stats,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


def print_building_result(r: BuildingResult) -> None:
    log.info(
        "  %s  id=%s  place=%s  return=%.1f  steps=%d",
        r.building_type,
        r.building_id,
        r.place,
        r.episode_return,
        r.n_steps,
    )
    log.info(
        "  %-35s %6s %6s %6s %6s  %8s %8s %8s",
        "Zone",
        "Mean",
        "Std",
        "Min",
        "Max",
        "<20°C",
        ">22°C",
        "InBand",
    )
    log.info("  " + "-" * 110)
    for zs in r.zone_stats:
        log.info(
            "  %-35s %6.1f %6.1f %6.1f %6.1f  %7.1f%% %7.1f%% %7.1f%%",
            zs.zone_name[:35],
            zs.mean,
            zs.std,
            zs.min,
            zs.max,
            zs.pct_below_heating,
            zs.pct_above_cooling,
            zs.pct_in_deadband,
        )


def print_type_summary(building_type: str, results: list[BuildingResult]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("SUMMARY: %s  (%d buildings)", building_type, len(results))
    log.info("=" * 80)

    all_stats: list[ZoneTemperatureStats] = []
    for r in results:
        all_stats.extend(r.zone_stats)

    if not all_stats:
        log.info("  No zone statistics available.")
        return

    log.info(
        "  %-35s %6s %6s %6s %6s  %8s %8s %8s",
        "Zone",
        "Mean",
        "Std",
        "Min",
        "Max",
        "<20°C",
        ">22°C",
        "InBand",
    )
    log.info("  " + "-" * 110)
    for zs in all_stats:
        log.info(
            "  %-35s %6.1f %6.1f %6.1f %6.1f  %7.1f%% %7.1f%% %7.1f%%",
            zs.zone_name[:35],
            zs.mean,
            zs.std,
            zs.min,
            zs.max,
            zs.pct_below_heating,
            zs.pct_above_cooling,
            zs.pct_in_deadband,
        )

    avg_in_band = np.mean([zs.pct_in_deadband for zs in all_stats])
    avg_below = np.mean([zs.pct_below_heating for zs in all_stats])
    avg_above = np.mean([zs.pct_above_cooling for zs in all_stats])
    worst_in_band = np.min([zs.pct_in_deadband for zs in all_stats])
    log.info("  " + "-" * 110)
    log.info(
        "  %-35s %6s %6s %6s %6s  %7.1f%% %7.1f%% %7.1f%%",
        "MEAN",
        "",
        "",
        "",
        "",
        avg_below,
        avg_above,
        avg_in_band,
    )
    log.info(
        "  %-35s %6s %6s %6s %6s  %7s  %7s  %7.1f%%",
        "WORST ZONE",
        "",
        "",
        "",
        "",
        "",
        "",
        worst_in_band,
    )
    avg_return = np.mean([r.episode_return for r in results])
    log.info("  Mean episode return: %.1f", avg_return)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-period",
        default="full_year",
        choices=["winter", "summer", "full_year"],
    )
    parser.add_argument("--n-buildings", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--types",
        nargs="+",
        default=["Warehouse", "RetailStandalone", "RestaurantFastFood"],
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path(
        tempfile.mkdtemp(prefix="validate_tuned_g36_")
    )
    log.info("Output directory: %s", output_dir)

    for btype in args.types:
        config_path = Path(_TYPE_CONFIGS[btype])
        if not config_path.exists():
            log.error(
                "Config %s not found — run tune_g36.py for %s first.",
                config_path,
                btype,
            )
            continue

        policy_cfg = load_policy_config(config_path)
        log.info("")
        log.info("=" * 80)
        log.info("Building type: %s  config: %s", btype, config_path)
        log.info("  params: %s", OmegaConf.to_container(policy_cfg, resolve=True))
        log.info("=" * 80)

        type_results: list[BuildingResult] = []
        for idx in range(args.n_buildings):
            try:
                r = run_one_building(
                    building_type=btype,
                    split_index=idx,
                    run_period=args.run_period,
                    policy_cfg=policy_cfg,
                    output_dir=output_dir,
                )
                type_results.append(r)
                print_building_result(r)
            except Exception:
                log.exception("FAILED for %s idx=%d", btype, idx)

        print_type_summary(btype, type_results)


if __name__ == "__main__":
    main()
