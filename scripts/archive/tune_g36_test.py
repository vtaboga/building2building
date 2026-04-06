#!/usr/bin/env python3
"""Optuna-based parameter tuning for the ASHRAE G36 controller on single-zone test buildings.

Tunes a single set of G36 parameters by rolling out trajectories on the first
N buildings of the ``action_space_2_zone_1`` test split and averaging the
in-band percentage across all of them.

Usage:
    python scripts/tune_g36_test.py --n-buildings 8 --n-trials 200
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import optuna
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

HEATING_SP = 20.25
COOLING_SP = 21.75

REWARD_BAND_LOW = 20.0
REWARD_BAND_HIGH = 22.0


# ── Temperature metric ───────────────────────────────────────────────────────


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
    low: float = REWARD_BAND_LOW,
    high: float = REWARD_BAND_HIGH,
) -> tuple[float, float]:
    """Return (mean_pct_in_band, worst_zone_pct_in_band)."""
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


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_params_single(
    split_index: int,
    params: dict[str, float],
    run_period: str,
    eplus_dir: Path,
) -> tuple[float, float]:
    """Rollout one building and return (mean_pct, worst_zone_pct)."""
    eplus_dir.mkdir(parents=True, exist_ok=True)

    policy_cfg = OmegaConf.create(
        {
            "heating_setpoint_c": HEATING_SP,
            "cooling_setpoint_c": COOLING_SP,
            **params,
            "availability_on": 2.0,
        }
    )

    env = make_single_zone_env(
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
        return compute_pct_in_band(data.obs, temp_indices)
    finally:
        try:
            env.close()
        except Exception:
            pass


def evaluate_params_multi(
    split_indices: list[int],
    params: dict[str, float],
    run_period: str,
    eplus_base_dir: Path,
) -> tuple[float, float]:
    """Rollout on multiple buildings, return (avg_mean_pct, avg_worst_pct)."""
    mean_pcts: list[float] = []
    worst_pcts: list[float] = []
    for idx in split_indices:
        eplus_dir = eplus_base_dir / f"bldg_{idx}"
        m, w = evaluate_params_single(
            split_index=idx,
            params=params,
            run_period=run_period,
            eplus_dir=eplus_dir,
        )
        mean_pcts.append(m)
        worst_pcts.append(w)
        log.info("  building idx=%d: mean=%.1f%%  worst=%.1f%%", idx, m, w)
    return float(np.mean(mean_pcts)), float(np.mean(worst_pcts))


# ── Optuna objective ──────────────────────────────────────────────────────────


def make_objective(
    split_indices: list[int],
    run_period: str,
    eplus_base_dir: Path,
):
    def objective(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("kp", 0.001, 0.1)
        ki = trial.suggest_float("ki", 0.0001, 0.02, log=True)
        integral_max = trial.suggest_float("integral_max", 5.0, 100.0)
        min_fan_fraction = trial.suggest_float("min_fan_fraction", 0.005, 0.10)
        sat_min_c = trial.suggest_float("sat_min_c", 6.0, 16.0)
        sat_max_c = trial.suggest_float("sat_max_c", 25.0, 60.0)
        sat_initial_c = trial.suggest_float("sat_initial_c", sat_min_c, sat_max_c)
        sat_trim = trial.suggest_float("sat_trim", 0.01, 1.5)
        sat_respond = trial.suggest_float("sat_respond", 0.5, 6.0)
        demand_deadband = trial.suggest_float("demand_deadband", 0.01, 1.0)

        params = {
            "kp": kp,
            "ki": ki,
            "integral_max": integral_max,
            "min_fan_fraction": min_fan_fraction,
            "sat_min_c": sat_min_c,
            "sat_max_c": sat_max_c,
            "sat_initial_c": sat_initial_c,
            "sat_trim": sat_trim,
            "sat_respond": sat_respond,
            "demand_deadband": demand_deadband,
        }

        trial_dir = eplus_base_dir / f"trial_{trial.number}"
        avg_mean_pct, avg_worst_pct = evaluate_params_multi(
            split_indices=split_indices,
            params=params,
            run_period=run_period,
            eplus_base_dir=trial_dir,
        )

        trial.set_user_attr("avg_worst_zone_pct", avg_worst_pct)
        log.info(
            "Trial %d: avg_mean_in_band=%.1f%%  avg_worst_zone=%.1f%%",
            trial.number,
            avg_mean_pct,
            avg_worst_pct,
        )
        return avg_mean_pct

    return objective


# ── Config I/O ────────────────────────────────────────────────────────────────


CONFIG_OUTPUT_PATH = Path("configs/policy/unitary_g36_singlezone_test.yaml")


def write_best_config(best_params: dict[str, float], output_path: Path) -> None:
    config = {
        "type": "unitary_g36",
        "heating_setpoint_c": HEATING_SP,
        "cooling_setpoint_c": COOLING_SP,
        "kp": round(float(best_params["kp"]), 4),
        "ki": round(float(best_params["ki"]), 6),
        "integral_max": round(float(best_params["integral_max"]), 1),
        "min_fan_fraction": round(float(best_params["min_fan_fraction"]), 4),
        "sat_min_c": round(float(best_params["sat_min_c"]), 2),
        "sat_max_c": round(float(best_params["sat_max_c"]), 2),
        "sat_initial_c": round(float(best_params["sat_initial_c"]), 2),
        "sat_trim": round(float(best_params["sat_trim"]), 3),
        "sat_respond": round(float(best_params["sat_respond"]), 3),
        "demand_deadband": round(float(best_params["demand_deadband"]), 3),
        "availability_on": 2.0,
        "target_schedule": {"enabled": False},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote best config to %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-buildings", type=int, default=8,
        help="Number of buildings from the start of the test split to evaluate on.",
    )
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument(
        "--run-period", default="full_year", choices=["winter", "summer", "full_year"]
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tune_g36_test"))
    args = parser.parse_args()

    split_indices = list(range(args.n_buildings))
    log.info(
        "Tuning G36 on %d single-zone test buildings (indices %s)",
        len(split_indices),
        split_indices,
    )

    study_dir = args.output_dir
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / "study.db"
    study_name = f"tune_g36_test_{args.n_buildings}bldg"

    eplus_base_dir = Path(
        tempfile.mkdtemp(prefix=f"tune_g36_test_{args.n_buildings}bldg_")
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
        split_indices=split_indices,
        run_period=args.run_period,
        eplus_base_dir=eplus_base_dir,
    )

    log.info(
        "Starting Optuna study %r (%d trials, period=%s)",
        study_name,
        args.n_trials,
        args.run_period,
    )
    study.optimize(objective, n_trials=args.n_trials)

    log.info("=" * 70)
    log.info("Best trial: #%d", study.best_trial.number)
    log.info("  avg_mean_in_band = %.2f%%", study.best_value)
    log.info(
        "  avg_worst_zone   = %.2f%%",
        study.best_trial.user_attrs.get("avg_worst_zone_pct", float("nan")),
    )
    log.info("  params: %s", study.best_params)

    write_best_config(dict(study.best_params), CONFIG_OUTPUT_PATH)


if __name__ == "__main__":
    main()
