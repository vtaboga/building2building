#!/usr/bin/env python3
"""Evaluate tuned G36 controllers on all test-split buildings.

For each building in the full test split, looks up its climate zone from the
metadata, loads the matching per-(building_type, climate_zone) YAML config,
and runs a full-year rollout.  Results are written to a CSV file.

Usage:
    python scripts/eval_g36_all_test.py --building-type Warehouse
    python scripts/eval_g36_all_test.py --building-type OfficeSmall --output-dir outputs/eval_g36
"""

from __future__ import annotations

import argparse
import csv
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf

from b2b.api import make_multizones_env
from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
from b2b.benchmark.runner import run_rollout
from b2b.sources.multizones_reference_buildings import (
    BuildingType,
    PLACE_TO_CLIMATE_ZONE,
    climate_zone_for_building,
    load_split_ids,
)
from b2b.types import RunPeriodConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEATING_SP = 20.0
COOLING_SP = 22.0

BUILDING_TYPES: list[BuildingType] = [
    "OfficeSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "Warehouse",
]


def config_path_for(building_type: BuildingType, climate_zone: int) -> Path:
    bt = building_type.lower()
    return Path(f"configs/policy/unitary_g36_{bt}_cz{climate_zone}.yaml")


def load_policy_config(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


def _zone_temp_indices(
    obs_names: list[str], controlled_zones: list[str]
) -> list[int]:
    prefix = "zone air temperature"
    indices: list[int] = []
    for zone in controlled_zones:
        zn = zone.strip().lower()
        for i, name in enumerate(obs_names):
            nl = name.strip().lower()
            if nl.startswith(prefix):
                zone_part = nl[len(prefix) :].strip()
                if zone_part == zn or zn in zone_part or zone_part in zn:
                    indices.append(i)
                    break
    return indices


def compute_pct_in_band(
    obs: np.ndarray,
    temp_indices: list[int],
    low: float = HEATING_SP,
    high: float = COOLING_SP,
) -> tuple[float, float]:
    if not temp_indices:
        return 0.0, 0.0
    pcts: list[float] = []
    for idx in temp_indices:
        temps = obs[:, idx]
        n = len(temps)
        if n == 0:
            continue
        pcts.append(float(np.sum((temps >= low) & (temps <= high)) / n * 100))
    if not pcts:
        return 0.0, 0.0
    return float(np.mean(pcts)), float(np.min(pcts))


CSV_FIELDS = [
    "building_type",
    "building_id",
    "split_index",
    "place",
    "climate_zone",
    "mean_pct_in_band",
    "worst_zone_pct_in_band",
    "episode_return",
    "n_steps",
]


def eval_one_building(
    building_type: BuildingType,
    split_index: int,
    building_id: int,
    climate_zone: int,
    policy_cfg: Any,
    run_period: str,
    eplus_dir: Path,
) -> dict[str, Any]:
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
        controlled_zones: list[str] = meta.get("controlled_zones", [])
        source = meta.get("building_source_metadata", {})
        temp_indices = _zone_temp_indices(obs_names, controlled_zones)

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

        mean_pct, worst_pct = compute_pct_in_band(data.obs, temp_indices)
        return {
            "building_type": building_type,
            "building_id": building_id,
            "split_index": split_index,
            "place": source.get("place", "?"),
            "climate_zone": climate_zone,
            "mean_pct_in_band": round(mean_pct, 2),
            "worst_zone_pct_in_band": round(worst_pct, 2),
            "episode_return": round(results[0].total_reward, 1),
            "n_steps": results[0].n_steps,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--building-type",
        required=True,
        choices=list(BUILDING_TYPES),
    )
    parser.add_argument(
        "--run-period", default="full_year", choices=["winter", "summer", "full_year"]
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_g36"))
    args = parser.parse_args()

    building_type: BuildingType = args.building_type  # type: ignore[assignment]
    output_dir: Path = args.output_dir / building_type
    output_dir.mkdir(parents=True, exist_ok=True)

    eplus_base_dir = Path(tempfile.mkdtemp(prefix=f"eval_g36_{building_type}_"))
    log.info("EnergyPlus scratch dir: %s", eplus_base_dir)

    # Pre-load all CZ configs
    cz_configs: dict[int, Any] = {}
    for cz in sorted(set(PLACE_TO_CLIMATE_ZONE.values())):
        cfg_path = config_path_for(building_type, cz)
        if cfg_path.exists():
            cz_configs[cz] = load_policy_config(cfg_path)
            log.info("Loaded config for CZ%d: %s", cz, cfg_path)
        else:
            log.warning("Missing config for CZ%d: %s — buildings in this CZ will be skipped", cz, cfg_path)

    test_ids = load_split_ids(building_type, "test")
    log.info("Evaluating %d test buildings for %s", len(test_ids), building_type)

    csv_path = output_dir / "results.csv"
    rows: list[dict[str, Any]] = []

    for idx, bid in enumerate(test_ids):
        cz = climate_zone_for_building(building_type, bid)
        if cz not in cz_configs:
            log.warning(
                "Skipping %s id=%d (CZ%d): no tuned config", building_type, bid, cz
            )
            continue

        log.info(
            "  [%d/%d] %s id=%d CZ%d", idx + 1, len(test_ids), building_type, bid, cz
        )
        try:
            row = eval_one_building(
                building_type=building_type,
                split_index=idx,
                building_id=bid,
                climate_zone=cz,
                policy_cfg=cz_configs[cz],
                run_period=args.run_period,
                eplus_dir=eplus_base_dir / f"idx{idx}",
            )
            rows.append(row)
            log.info(
                "    mean_in_band=%.1f%%  worst_zone=%.1f%%  return=%.1f",
                row["mean_pct_in_band"],
                row["worst_zone_pct_in_band"],
                row["episode_return"],
            )
        except Exception:
            log.exception("FAILED for %s id=%d idx=%d", building_type, bid, idx)

    # Write CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d results to %s", len(rows), csv_path)

    # Summary
    if rows:
        means = [r["mean_pct_in_band"] for r in rows]
        worsts = [r["worst_zone_pct_in_band"] for r in rows]
        log.info("=" * 70)
        log.info(
            "SUMMARY %s: %d buildings, mean_in_band=%.1f%%, worst_zone=%.1f%%",
            building_type,
            len(rows),
            float(np.mean(means)),
            float(np.min(worsts)),
        )
        for cz in sorted(set(r["climate_zone"] for r in rows)):
            cz_rows = [r for r in rows if r["climate_zone"] == cz]
            cz_means = [r["mean_pct_in_band"] for r in cz_rows]
            log.info(
                "  CZ%d: %d buildings, mean_in_band=%.1f%%",
                cz,
                len(cz_rows),
                float(np.mean(cz_means)),
            )


if __name__ == "__main__":
    main()
