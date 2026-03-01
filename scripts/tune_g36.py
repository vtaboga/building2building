#!/usr/bin/env python3
"""Optuna-based per-building-type parameter tuning for the ASHRAE G36 controller.

For each trial, instantiates a UnitaryG36Policy with suggested parameters,
evaluates on N test-split buildings, and maximises the mean % of timesteps
where zone temperatures fall within [20, 22]°C.

Usage:
    python scripts/tune_g36.py --building-type Warehouse --n-trials 50
    python scripts/tune_g36.py --building-type RetailStandalone --n-trials 50
    python scripts/tune_g36.py --building-type RestaurantFastFood --n-trials 50
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import yaml
from omegaconf import OmegaConf

from b2b.api import make_multizones_env
from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
from b2b.benchmark.runner import run_rollout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEATING_SP = 20.0
COOLING_SP = 22.0

BuildingType = Literal["OfficeSmall"]


def _zone_temp_indices(
    obs_names: list[str], controlled_zones: list[str]
) -> list[int]:
    """Return obs column indices for each controlled zone's air temperature."""
    prefix = "zone air temperature"
    indices: list[int] = []
    for zone in controlled_zones:
        zn = zone.strip().lower()
        found = False
        for i, name in enumerate(obs_names):
            nl = name.strip().lower()
            if nl.startswith(prefix):
                zone_part = nl[len(prefix) :].strip()
                if zone_part == zn or zn in zone_part or zone_part in zn:
                    indices.append(i)
                    found = True
                    break
        if not found:
            log.warning("Could not find temperature obs for zone %r", zone)
    return indices


def compute_pct_in_band(
    obs: np.ndarray,
    temp_indices: list[int],
    low: float = HEATING_SP,
    high: float = COOLING_SP,
) -> tuple[float, float]:
    """Return (mean_pct_in_band, worst_zone_pct_in_band) across zones."""
    if not temp_indices:
        return 0.0, 0.0
    pcts: list[float] = []
    for idx in temp_indices:
        temps = obs[:, idx]
        n = len(temps)
        if n == 0:
            continue
        in_band = float(np.sum((temps >= low) & (temps <= high)) / n * 100)
        pcts.append(in_band)
    if not pcts:
        return 0.0, 0.0
    return float(np.mean(pcts)), float(np.min(pcts))


def evaluate_params(
    building_type: str,
    params: dict[str, float],
    n_buildings: int,
    run_period: str,
    eplus_base_dir: Path,
) -> tuple[float, float]:
    """Run rollouts on N buildings and return (mean_pct, worst_zone_pct)."""
    all_zone_pcts: list[float] = []
    worst_zones: list[float] = []

    policy_cfg = OmegaConf.create(
        {
            "heating_setpoint_c": HEATING_SP,
            "cooling_setpoint_c": COOLING_SP,
            "kp": params["kp"],
            "ki": params["ki"],
            "min_fan_fraction": params["min_fan_fraction"],
            "med_fan_fraction": params["med_fan_fraction"],
            "sat_min_c": params["sat_min_c"],
            "sat_max_c": params.get("sat_max_c"),
            "availability_on": 2.0,
        }
    )

    for idx in range(n_buildings):
        eplus_dir = eplus_base_dir / f"{building_type}_idx{idx}"
        eplus_dir.mkdir(parents=True, exist_ok=True)

        env = make_multizones_env(
            building_type=building_type,
            split="test",
            split_index=idx,
            eplus_output_dir=str(eplus_dir),
            task={"run_period": run_period},
        )
        try:
            meta = env.metadata
            obs_names: list[str] = meta["observation_names"]
            controlled_zones: list[str] = meta.get("controlled_zones", [])
            temp_indices = _zone_temp_indices(obs_names, controlled_zones)

            policy = UnitaryG36Policy(policy_cfg)
            max_steps = (
                env.spec.max_episode_steps
                if env.spec and env.spec.max_episode_steps
                else 35040
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
            all_zone_pcts.append(mean_pct)
            worst_zones.append(worst_pct)

            source = meta.get("building_source_metadata", {})
            log.info(
                "  [%s idx=%d id=%s] mean_in_band=%.1f%%  worst_zone=%.1f%%  return=%.1f",
                building_type,
                idx,
                source.get("building_id", "?"),
                mean_pct,
                worst_pct,
                results[0].total_reward,
            )
        finally:
            try:
                env.close()
            except Exception:
                pass

    if not all_zone_pcts:
        return 0.0, 0.0
    return float(np.mean(all_zone_pcts)), float(np.min(worst_zones))


def make_objective(
    building_type: str,
    n_buildings: int,
    run_period: str,
    eplus_base_dir: Path,
) -> optuna.Study:
    """Create an Optuna objective closure."""

    def objective(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("kp", 0.5, 5.0)
        ki = trial.suggest_float("ki", 0.01, 0.5, log=True)
        min_fan_fraction = trial.suggest_float("min_fan_fraction", 0.15, 0.90)
        med_fan_fraction = trial.suggest_float(
            "med_fan_fraction", min_fan_fraction, 1.0
        )
        sat_min_c = trial.suggest_float("sat_min_c", 10.0, 18.0)
        sat_max_raw = trial.suggest_float("sat_max_c", 35.0, 80.0)
        use_null_sat_max = trial.suggest_categorical("sat_max_null", [True, False])
        sat_max_c: float | None = None if use_null_sat_max else sat_max_raw

        params = {
            "kp": kp,
            "ki": ki,
            "min_fan_fraction": min_fan_fraction,
            "med_fan_fraction": med_fan_fraction,
            "sat_min_c": sat_min_c,
            "sat_max_c": sat_max_c,
        }

        trial_dir = eplus_base_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        mean_pct, worst_pct = evaluate_params(
            building_type=building_type,
            params=params,
            n_buildings=n_buildings,
            run_period=run_period,
            eplus_base_dir=trial_dir,
        )

        trial.set_user_attr("worst_zone_pct", worst_pct)
        log.info(
            "Trial %d: mean_in_band=%.1f%%  worst_zone=%.1f%%  params=%s",
            trial.number,
            mean_pct,
            worst_pct,
            {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in params.items()},
        )
        return mean_pct

    return objective


def write_best_config(
    best_params: dict[str, float | bool | None],
    output_path: Path,
) -> None:
    """Write the best trial parameters to a YAML config file."""
    sat_max_null = best_params.pop("sat_max_null", False)
    sat_max_val = best_params.pop("sat_max_c", None)
    sat_max: float | None = None if sat_max_null else sat_max_val

    config = {
        "type": "unitary_g36",
        "heating_setpoint_c": HEATING_SP,
        "cooling_setpoint_c": COOLING_SP,
        "kp": round(float(best_params["kp"]), 4),
        "ki": round(float(best_params["ki"]), 6),
        "min_fan_fraction": round(float(best_params["min_fan_fraction"]), 4),
        "med_fan_fraction": round(float(best_params["med_fan_fraction"]), 4),
        "sat_min_c": round(float(best_params["sat_min_c"]), 2),
        "sat_max_c": round(sat_max, 2) if sat_max is not None else None,
        "availability_on": 2.0,
        "target_schedule": {"enabled": False},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote best config to %s", output_path)


_TYPE_TO_CONFIG_SUFFIX: dict[str, str] = {
    "Warehouse": "warehouse",
    "RetailStandalone": "retail",
    "RestaurantFastFood": "restaurant",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--building-type",
        required=True,
        choices=["Warehouse", "RetailStandalone", "RestaurantFastFood", "OfficeSmall"],
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--run-period",
        default="full_year",
        choices=["winter", "summer", "full_year"],
    )
    parser.add_argument("--n-buildings", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tune_g36"),
    )
    args = parser.parse_args()

    study_dir = args.output_dir / args.building_type
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / "study.db"
    study_name = f"tune_g36_{args.building_type}"

    eplus_base_dir = Path(
        tempfile.mkdtemp(prefix=f"tune_g36_{args.building_type}_")
    )
    log.info("EnergyPlus scratch dir: %s", eplus_base_dir)

    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = make_objective(
        building_type=args.building_type,
        n_buildings=args.n_buildings,
        run_period=args.run_period,
        eplus_base_dir=eplus_base_dir,
    )

    log.info(
        "Starting Optuna study %r (%d trials, %d buildings, period=%s)",
        study_name,
        args.n_trials,
        args.n_buildings,
        args.run_period,
    )
    study.optimize(objective, n_trials=args.n_trials)

    log.info("=" * 70)
    log.info("Best trial: #%d", study.best_trial.number)
    log.info("  mean_in_band = %.2f%%", study.best_value)
    log.info(
        "  worst_zone   = %.2f%%",
        study.best_trial.user_attrs.get("worst_zone_pct", float("nan")),
    )
    log.info("  params: %s", study.best_params)

    suffix = _TYPE_TO_CONFIG_SUFFIX.get(args.building_type, args.building_type.lower())
    config_path = Path(f"configs/policy/unitary_g36_{suffix}.yaml")
    write_best_config(dict(study.best_params), config_path)


if __name__ == "__main__":
    main()
