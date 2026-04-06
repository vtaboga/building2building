#!/usr/bin/env python3
"""Evaluate tuned ASHRAE AirLoop controllers on all OfficeMedium test buildings.

Uses per-climate-zone configs (ashrae_air_loop_officemedium_cz{1..8}.yaml).
Buildings whose CZ config is missing are skipped with a warning.

Usage:
    python scripts/eval_officemedium_test.py --output-dir outputs/eval_g36
"""

from __future__ import annotations

import argparse
import csv
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf

from building2building.api import make_multizones_env
from building2building.baselines.controllers.ashrae_air_loop import AshraeAirLoopPolicy
from building2building.benchmark.runner import run_rollout
from building2building.sources.multizones_reference_buildings import (
    PLACE_TO_CLIMATE_ZONE,
    climate_zone_for_building,
    load_split_ids,
)
from building2building.types import RunPeriodConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BUILDING_TYPE = "OfficeMedium"
HEATING_SP = 20.0
COOLING_SP = 22.0


@dataclass(frozen=True)
class EvalSetup:
    name: str
    reward_config_path: Path
    task_overrides: dict[str, Any]


EVAL_SETUPS: list[EvalSetup] = [
    EvalSetup(
        name="deadband_ew001_const",
        reward_config_path=Path("configs/reward/deadband_ew001.yaml"),
        task_overrides={"target_temperature_mode": "constant"},
    ),
    EvalSetup(
        name="deadband_ew01_const",
        reward_config_path=Path("configs/reward/deadband_ew01.yaml"),
        task_overrides={"target_temperature_mode": "constant"},
    ),
    EvalSetup(
        name="deadband_ew001_occ",
        reward_config_path=Path("configs/reward/deadband_ew001_occ.yaml"),
        task_overrides={
            "target_temperature_mode": "occupancy",
            "default_zone_target_temperature": {
                "occupied_c": 21.0,
                "unoccupied_c": 18.0,
            },
        },
    ),
    EvalSetup(
        name="barrier_ew1",
        reward_config_path=Path("configs/reward/barrier_ew1.yaml"),
        task_overrides={},
    ),
]


def config_path_for(climate_zone: int) -> Path:
    return Path(f"configs/policy/ashrae_air_loop_officemedium_cz{climate_zone}.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def load_policy_config(path: Path) -> Any:
    return OmegaConf.create(load_yaml(path))


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


def _zone_target_temp_indices(
    obs_names: list[str], controlled_zones: list[str]
) -> list[int]:
    prefix = "target_temperature"
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
    target_indices: list[int] | None = None,
    dT: float = 1.0,
) -> tuple[float, float]:
    if not temp_indices:
        return 0.0, 0.0
    pcts: list[float] = []
    for zi, idx in enumerate(temp_indices):
        temps = obs[:, idx]
        n = len(temps)
        if n == 0:
            continue
        if target_indices and zi < len(target_indices):
            targets = obs[:, target_indices[zi]]
            in_band = (temps >= targets - dT) & (temps <= targets + dT)
        else:
            in_band = (temps >= low) & (temps <= high)
        pcts.append(float(np.sum(in_band) / n * 100))
    if not pcts:
        return 0.0, 0.0
    return float(np.mean(pcts)), float(np.min(pcts))


CSV_FIELDS = [
    "setup",
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
    split_index: int,
    building_id: int,
    climate_zone: int,
    policy_cfg: Any,
    run_period: str,
    eplus_dir: Path,
    reward: dict[str, Any],
    task_overrides: dict[str, Any],
) -> dict[str, Any]:
    eplus_dir.mkdir(parents=True, exist_ok=True)
    task_dict: dict[str, Any] = {"run_period": run_period, **task_overrides}
    env = make_multizones_env(
        building_type=BUILDING_TYPE,
        split="test",
        split_index=split_index,
        eplus_output_dir=str(eplus_dir),
        task=task_dict,
        reward=reward,
    )
    try:
        meta = env.metadata
        obs_names: list[str] = meta["observation_names"]
        controlled_zones: list[str] = meta.get("controlled_zones", [])
        source = meta.get("building_source_metadata", {})
        temp_indices = _zone_temp_indices(obs_names, controlled_zones)
        target_indices = _zone_target_temp_indices(obs_names, controlled_zones)

        policy = AshraeAirLoopPolicy(policy_cfg)
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

        mean_pct, worst_pct = compute_pct_in_band(
            data.obs,
            temp_indices,
            target_indices=target_indices or None,
        )
        return {
            "building_type": BUILDING_TYPE,
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


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a single row to *csv_path*, writing the header if the file is new."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_setup(
    setup: EvalSetup,
    run_period: str,
    output_dir: Path,
    eplus_base_dir: Path,
    cz_configs: dict[int, Any],
    test_ids: list[int],
) -> None:
    reward_dict = load_yaml(setup.reward_config_path)
    setup_dir = output_dir / setup.name
    setup_dir.mkdir(parents=True, exist_ok=True)

    log.info("--- Setup: %s  reward=%s ---", setup.name, setup.reward_config_path)

    csv_path = setup_dir / "results.csv"
    if csv_path.exists():
        csv_path.unlink()

    n_ok = 0
    all_mean_pcts: list[float] = []
    all_worst_pcts: list[float] = []

    for idx, bid in enumerate(test_ids):
        cz = climate_zone_for_building(BUILDING_TYPE, bid)
        if cz not in cz_configs:
            log.warning(
                "Skipping %s id=%d (CZ%d): no tuned config",
                BUILDING_TYPE, bid, cz,
            )
            continue

        log.info(
            "  [%d/%d] %s id=%d CZ%d  setup=%s",
            idx + 1, len(test_ids), BUILDING_TYPE, bid, cz, setup.name,
        )
        try:
            row = eval_one_building(
                split_index=idx,
                building_id=bid,
                climate_zone=cz,
                policy_cfg=cz_configs[cz],
                run_period=run_period,
                eplus_dir=eplus_base_dir / setup.name / f"idx{idx}",
                reward=reward_dict,
                task_overrides=setup.task_overrides,
            )
            row["setup"] = setup.name
            _append_row(csv_path, row)
            n_ok += 1
            all_mean_pcts.append(row["mean_pct_in_band"])
            all_worst_pcts.append(row["worst_zone_pct_in_band"])
            log.info(
                "    mean_in_band=%.1f%%  worst_zone=%.1f%%  return=%.1f",
                row["mean_pct_in_band"],
                row["worst_zone_pct_in_band"],
                row["episode_return"],
            )
        except Exception:
            log.exception(
                "FAILED for %s id=%d idx=%d setup=%s",
                BUILDING_TYPE, bid, idx, setup.name,
            )

    log.info("Wrote %d results to %s", n_ok, csv_path)

    if all_mean_pcts:
        log.info(
            "SUMMARY %s / %s: %d buildings, mean_in_band=%.1f%%, worst_zone=%.1f%%",
            BUILDING_TYPE, setup.name, n_ok,
            float(np.mean(all_mean_pcts)), float(np.min(all_worst_pcts)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-period", default="full_year",
        choices=["winter", "summer", "full_year"],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/eval_g36"),
    )
    parser.add_argument(
        "--setup",
        choices=[s.name for s in EVAL_SETUPS],
        default=None,
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir / BUILDING_TYPE
    output_dir.mkdir(parents=True, exist_ok=True)

    eplus_base_dir = Path(tempfile.mkdtemp(prefix=f"eval_{BUILDING_TYPE}_"))
    log.info("EnergyPlus scratch dir: %s", eplus_base_dir)

    cz_configs: dict[int, Any] = {}
    for cz in sorted(set(PLACE_TO_CLIMATE_ZONE.values())):
        cfg_path = config_path_for(cz)
        if cfg_path.exists():
            cz_configs[cz] = load_policy_config(cfg_path)
            log.info("Loaded config for CZ%d: %s", cz, cfg_path)
        else:
            log.warning(
                "Missing config for CZ%d: %s — buildings in this CZ will be skipped",
                cz, cfg_path,
            )

    test_ids = load_split_ids(BUILDING_TYPE, "test")
    log.info("Evaluating %d test buildings for %s", len(test_ids), BUILDING_TYPE)

    setups = EVAL_SETUPS
    if args.setup is not None:
        setups = [s for s in EVAL_SETUPS if s.name == args.setup]

    for setup in setups:
        run_setup(
            setup=setup,
            run_period=args.run_period,
            output_dir=output_dir,
            eplus_base_dir=eplus_base_dir,
            cz_configs=cz_configs,
            test_ids=test_ids,
        )

    log.info("=" * 70)
    log.info("All setups complete for %s.", BUILDING_TYPE)


if __name__ == "__main__":
    main()
