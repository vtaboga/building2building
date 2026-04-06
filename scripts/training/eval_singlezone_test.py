#!/usr/bin/env python3
"""Evaluate tuned G36 controllers on all single-zone test buildings.

Uses a single tuned G36 config for all buildings (no per-CZ configs).
Buildings that fail (e.g. action space mismatch) are skipped with a warning.

Usage:
    python scripts/eval_singlezone_test.py --output-dir outputs/eval_g36
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf

from building2building.api import make_single_zone_env
from building2building.baselines.controllers.unitary_g36 import UnitaryG36Policy
from building2building.benchmark.runner import run_rollout
from building2building.types import RunPeriodConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEATING_SP = 20.0
COOLING_SP = 22.0

POLICY_CONFIG_PATH = Path("configs/policy/unitary_g36_singlezone_test.yaml")
SPLIT_DATA_PATH = Path("b2b/sources/data/action_space_2_zone_1_test_data.json")


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def load_test_ids() -> list[int]:
    return json.loads(SPLIT_DATA_PATH.read_text(encoding="utf-8"))


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
    policy_cfg: Any,
    run_period: str,
    eplus_dir: Path,
    reward: dict[str, Any],
    task_overrides: dict[str, Any],
) -> dict[str, Any]:
    eplus_dir.mkdir(parents=True, exist_ok=True)
    task_dict: dict[str, Any] = {"run_period": run_period, **task_overrides}
    env = make_single_zone_env(
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
        temp_indices = _zone_temp_indices(obs_names, controlled_zones)
        target_indices = _zone_target_temp_indices(obs_names, controlled_zones)

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

        mean_pct, worst_pct = compute_pct_in_band(
            data.obs,
            temp_indices,
            target_indices=target_indices or None,
        )
        return {
            "building_type": "SingleZone",
            "building_id": building_id,
            "split_index": split_index,
            "place": "?",
            "climate_zone": 0,
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


def run_setup(
    setup: EvalSetup,
    policy_cfg: Any,
    run_period: str,
    output_dir: Path,
    eplus_base_dir: Path,
    test_ids: list[int],
) -> tuple[int, int]:
    """Run one setup across all test buildings. Returns (success, skipped)."""
    reward_dict = load_yaml(setup.reward_config_path)
    setup_dir = output_dir / setup.name
    setup_dir.mkdir(parents=True, exist_ok=True)

    log.info("--- Setup: %s  reward=%s ---", setup.name, setup.reward_config_path)

    csv_path = setup_dir / "results.csv"
    rows: list[dict[str, Any]] = []
    skipped = 0

    for idx, bid in enumerate(test_ids):
        log.info(
            "  [%d/%d] SingleZone id=%d  setup=%s",
            idx + 1, len(test_ids), bid, setup.name,
        )
        try:
            row = eval_one_building(
                split_index=idx,
                building_id=bid,
                policy_cfg=policy_cfg,
                run_period=run_period,
                eplus_dir=eplus_base_dir / setup.name / f"idx{idx}",
                reward=reward_dict,
                task_overrides=setup.task_overrides,
            )
            row["setup"] = setup.name
            rows.append(row)
            log.info(
                "    mean_in_band=%.1f%%  worst_zone=%.1f%%  return=%.1f",
                row["mean_pct_in_band"],
                row["worst_zone_pct_in_band"],
                row["episode_return"],
            )
        except Exception:
            skipped += 1
            log.warning(
                "SKIPPED SingleZone id=%d idx=%d setup=%s (action space mismatch or other error)",
                bid, idx, setup.name,
                exc_info=True,
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d results to %s (skipped %d)", len(rows), csv_path, skipped)
    return len(rows), skipped


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

    output_dir: Path = args.output_dir / "SingleZone"
    output_dir.mkdir(parents=True, exist_ok=True)

    eplus_base_dir = Path(tempfile.mkdtemp(prefix="eval_singlezone_"))
    log.info("EnergyPlus scratch dir: %s", eplus_base_dir)

    policy_cfg = OmegaConf.create(load_yaml(POLICY_CONFIG_PATH))
    log.info("Loaded policy config: %s", POLICY_CONFIG_PATH)

    test_ids = load_test_ids()
    log.info("Evaluating %d single-zone test buildings", len(test_ids))

    setups = EVAL_SETUPS
    if args.setup is not None:
        setups = [s for s in EVAL_SETUPS if s.name == args.setup]

    total_ok = 0
    total_skip = 0
    for setup in setups:
        ok, skip = run_setup(
            setup=setup,
            policy_cfg=policy_cfg,
            run_period=args.run_period,
            output_dir=output_dir,
            eplus_base_dir=eplus_base_dir,
            test_ids=test_ids,
        )
        total_ok += ok
        total_skip += skip

    log.info("=" * 70)
    log.info(
        "All setups complete for SingleZone. Success: %d, Skipped: %d",
        total_ok, total_skip,
    )


if __name__ == "__main__":
    main()
