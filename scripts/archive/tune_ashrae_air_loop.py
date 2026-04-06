#!/usr/bin/env python3
"""Optuna-based parameter tuning for the ASHRAE air-loop controller.

Tunes one set of parameters per climate zone for the OfficeMedium building type.
Each climate zone has exactly one building in the ``test_small`` split;
that single building is used for evaluation during optimisation.

Usage:
    python scripts/tune_ashrae_air_loop.py --climate-zone 3 --n-trials 200
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

from building2building.api import make_multizones_env
from building2building.baselines.controllers.ashrae_air_loop import AshraeAirLoopPolicy
from building2building.benchmark.runner import run_rollout
from building2building.sources.multizones_reference_buildings import (
    BuildingType,
    CLIMATE_ZONES,
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

BUILDING_TYPE: BuildingType = "OfficeMedium"

REWARD_BAND_LOW = 20.0
REWARD_BAND_HIGH = 22.0


def test_small_index_for_climate_zone(climate_zone: int) -> int:
    """Return the test_small split index whose building belongs to *climate_zone*."""
    ids = load_split_ids(BUILDING_TYPE, "test_small")
    for idx, bid in enumerate(ids):
        if climate_zone_for_building(BUILDING_TYPE, bid) == climate_zone:
            return idx
    raise ValueError(
        f"No test_small building for {BUILDING_TYPE} in CZ {climate_zone}"
    )


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


def evaluate_params(
    split: str,
    split_index: int,
    params: dict[str, float],
    run_period: str,
    eplus_dir: Path,
) -> tuple[float, float]:
    """Rollout one building and return (mean_pct, worst_zone_pct)."""
    eplus_dir.mkdir(parents=True, exist_ok=True)

    policy_cfg = OmegaConf.create(params)

    env = make_multizones_env(
        building_type=BUILDING_TYPE,
        split=split,
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
        return compute_pct_in_band(data.obs, temp_indices)
    finally:
        try:
            env.close()
        except Exception:
            pass


# ── Optuna objective ──────────────────────────────────────────────────────────


def make_objective(
    split: str,
    split_index: int,
    run_period: str,
    eplus_base_dir: Path,
):
    def objective(trial: optuna.Trial) -> float:
        target_temp = trial.suggest_float("target_temp", 20.0, 22.0)
        deadband = trial.suggest_float("deadband", 0.3, 2.0)

        sat_neutral = trial.suggest_float("sat_neutral", 16.0, 25.0)
        sat_kp = trial.suggest_float("sat_kp", 0.1, 3.0)
        sat_min = trial.suggest_float("sat_min", 6.0, 16.0)
        sat_max = trial.suggest_float("sat_max", 30.0, 60.0)
        sat_rate_limit = trial.suggest_float("sat_rate_limit", 0.05, 1.0)
        outdoor_sat_gain = trial.suggest_float("outdoor_sat_gain", 0.0, 0.3)
        sat_cold_bias = trial.suggest_float("sat_cold_bias", 0.0, 0.5)
        sat_warm_bias = trial.suggest_float("sat_warm_bias", 0.0, 1.0)

        flow_base = trial.suggest_float("flow_base", 0.2, 0.7)
        flow_kp = trial.suggest_float("flow_kp", 0.05, 0.6)
        flow_ki = trial.suggest_float("flow_ki", 0.005, 0.1, log=True)
        flow_min = trial.suggest_float("flow_min", 0.05, 0.3)
        flow_max = trial.suggest_float("flow_max", 0.7, 1.0)
        flow_rate_limit = trial.suggest_float("flow_rate_limit", 0.01, 0.15)
        integral_max = trial.suggest_float("integral_max", 5.0, 50.0)
        integral_decay = trial.suggest_float("integral_decay", 0.9, 1.0)

        reheat_sp_min = trial.suggest_float("reheat_sp_min", 8.0, 15.0)
        reheat_sp_max = trial.suggest_float("reheat_sp_max", 20.0, 30.0)
        reheat_sp_deadband = trial.suggest_float("reheat_sp_deadband", 0.1, 1.0)
        reheat_sp_kp = trial.suggest_float("reheat_sp_kp", 1.0, 6.0)
        reheat_sp_rate_limit = trial.suggest_float("reheat_sp_rate_limit", 0.05, 0.5)

        clg_sp_default = trial.suggest_float("clg_sp_default", 21.0, 26.0)
        error_ema_alpha = trial.suggest_float("error_ema_alpha", 0.1, 0.6)

        params = {
            "target_temp": target_temp,
            "deadband": deadband,
            "sat_neutral": sat_neutral,
            "sat_kp": sat_kp,
            "sat_min": sat_min,
            "sat_max": sat_max,
            "sat_rate_limit": sat_rate_limit,
            "outdoor_sat_gain": outdoor_sat_gain,
            "sat_cold_bias": sat_cold_bias,
            "sat_warm_bias": sat_warm_bias,
            "flow_base": flow_base,
            "flow_kp": flow_kp,
            "flow_ki": flow_ki,
            "flow_min": flow_min,
            "flow_max": flow_max,
            "flow_rate_limit": flow_rate_limit,
            "integral_max": integral_max,
            "integral_decay": integral_decay,
            "reheat_sp_min": reheat_sp_min,
            "reheat_sp_max": reheat_sp_max,
            "reheat_sp_deadband": reheat_sp_deadband,
            "reheat_sp_kp": reheat_sp_kp,
            "reheat_sp_rate_limit": reheat_sp_rate_limit,
            "clg_sp_default": clg_sp_default,
            "error_ema_alpha": error_ema_alpha,
        }

        trial_dir = eplus_base_dir / f"trial_{trial.number}"
        mean_pct, worst_pct = evaluate_params(
            split=split,
            split_index=split_index,
            params=params,
            run_period=run_period,
            eplus_dir=trial_dir,
        )

        trial.set_user_attr("worst_zone_pct", worst_pct)
        log.info(
            "Trial %d: mean_in_band=%.1f%%  worst_zone=%.1f%%",
            trial.number,
            mean_pct,
            worst_pct,
        )
        return mean_pct

    return objective


# ── Config I/O ────────────────────────────────────────────────────────────────


def config_path_for(climate_zone: int) -> Path:
    return Path(f"configs/policy/ashrae_air_loop_officemedium_cz{climate_zone}.yaml")


def write_best_config(best_params: dict[str, float], output_path: Path) -> None:
    config: dict[str, object] = {"type": "ashrae_air_loop"}
    for key, value in best_params.items():
        config[key] = round(float(value), 4)
    config["sat_aware_flow"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote best config to %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climate-zone", type=int, required=True, choices=CLIMATE_ZONES)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument(
        "--run-period", default="full_year", choices=["winter", "summer", "full_year"]
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tune_ashrae_air_loop"))
    args = parser.parse_args()

    cz: int = args.climate_zone

    split_index = test_small_index_for_climate_zone(cz)
    log.info(
        "Tuning %s CZ%d  (test_small split_index=%d)", BUILDING_TYPE, cz, split_index
    )

    study_dir = args.output_dir / f"cz{cz}"
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / "study.db"
    study_name = f"tune_ashrae_air_loop_officemedium_cz{cz}"

    eplus_base_dir = Path(
        tempfile.mkdtemp(prefix=f"tune_ashrae_air_loop_cz{cz}_")
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
        split="test_small",
        split_index=split_index,
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
    log.info("  mean_in_band = %.2f%%", study.best_value)
    log.info(
        "  worst_zone   = %.2f%%",
        study.best_trial.user_attrs.get("worst_zone_pct", float("nan")),
    )
    log.info("  params: %s", study.best_params)

    out_path = config_path_for(cz)
    write_best_config(dict(study.best_params), out_path)


if __name__ == "__main__":
    main()
