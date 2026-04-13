#!/usr/bin/env python3
"""Tune rule-based controller parameters with Optuna.

Supports both unitary HVAC (single-zone packaged systems) and
air-loop (VAV multi-zone systems) controllers. Optimizes the total
episode reward on a single building.

Usage with Hydra::

    python -m baselines.tune_controller experiment=tune_controller \
        building_type=OfficeSmall climate_zone=1

    python -m baselines.tune_controller experiment=tune_controller \
        building_type=OfficeMedium climate_zone=3 n_trials=100
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import hydra
import numpy as np
import optuna
import yaml
from omegaconf import DictConfig

import building2building as b2b
from baselines.controllers.air_loop import (
    AirLoopConfig,
    AirLoopPolicy,
)
from baselines.controllers.unitary_hvac import UnitaryHvacConfig, UnitaryHvacPolicy
from baselines.utils.evaluation import run_episode

logger = logging.getLogger(__name__)

VAV_BUILDING_TYPES = {"OfficeMedium"}


def _suggest_unitary_hvac(trial: optuna.Trial) -> UnitaryHvacConfig:
    return UnitaryHvacConfig(
        heating_setpoint_c=trial.suggest_float("heating_setpoint_c", 18.0, 22.0),
        cooling_setpoint_c=trial.suggest_float("cooling_setpoint_c", 20.0, 26.0),
        kp=trial.suggest_float("kp", 0.01, 3.0, log=True),
        ki=trial.suggest_float("ki", 1e-4, 0.1, log=True),
        integral_max=trial.suggest_float("integral_max", 1.0, 50.0),
        min_fan_fraction=trial.suggest_float("min_fan_fraction", 0.01, 0.5),
        sat_min_c=trial.suggest_float("sat_min_c", 5.0, 15.0),
        sat_max_c=trial.suggest_float("sat_max_c", 20.0, 55.0),
        sat_initial_c=trial.suggest_float("sat_initial_c", 10.0, 25.0),
        sat_trim=trial.suggest_float("sat_trim", 0.05, 1.0),
        sat_respond=trial.suggest_float("sat_respond", 0.1, 5.0),
        demand_deadband=trial.suggest_float("demand_deadband", 0.01, 2.0),
        availability_on=trial.suggest_float("availability_on", 0.5, 3.0),
    )


def _suggest_air_loop(trial: optuna.Trial) -> AirLoopConfig:
    return AirLoopConfig(
        target_temp=trial.suggest_float("target_temp", 18.0, 24.0),
        deadband=trial.suggest_float("deadband", 0.5, 3.0),
        sat_neutral=trial.suggest_float("sat_neutral", 14.0, 24.0),
        sat_kp=trial.suggest_float("sat_kp", 0.1, 5.0),
        sat_min=trial.suggest_float("sat_min", 5.0, 15.0),
        sat_max=trial.suggest_float("sat_max", 35.0, 65.0),
        sat_rate_limit=trial.suggest_float("sat_rate_limit", 0.05, 1.0),
        outdoor_sat_gain=trial.suggest_float("outdoor_sat_gain", 0.0, 0.5),
        sat_cold_bias=trial.suggest_float("sat_cold_bias", 0.0, 1.0),
        sat_warm_bias=trial.suggest_float("sat_warm_bias", 0.0, 1.5),
        flow_base=trial.suggest_float("flow_base", 0.1, 0.8),
        flow_kp=trial.suggest_float("flow_kp", 0.05, 1.0),
        flow_ki=trial.suggest_float("flow_ki", 1e-3, 0.1, log=True),
        flow_min=trial.suggest_float("flow_min", 0.05, 0.5),
        flow_max=trial.suggest_float("flow_max", 0.5, 1.0),
        flow_rate_limit=trial.suggest_float("flow_rate_limit", 0.01, 0.3),
        integral_max=trial.suggest_float("integral_max", 5.0, 50.0),
        integral_decay=trial.suggest_float("integral_decay", 0.8, 1.0),
        reheat_sp_min=trial.suggest_float("reheat_sp_min", 8.0, 18.0),
        reheat_sp_max=trial.suggest_float("reheat_sp_max", 20.0, 30.0),
        reheat_sp_deadband=trial.suggest_float("reheat_sp_deadband", 0.1, 2.0),
        reheat_sp_kp=trial.suggest_float("reheat_sp_kp", 0.5, 5.0),
        reheat_sp_rate_limit=trial.suggest_float("reheat_sp_rate_limit", 0.02, 0.3),
        clg_sp_default=trial.suggest_float("clg_sp_default", 20.0, 28.0),
        error_ema_alpha=trial.suggest_float("error_ema_alpha", 0.05, 0.5),
        sat_aware_flow=trial.suggest_categorical("sat_aware_flow", [True, False]),
    )


def _config_to_dict(cfg: UnitaryHvacConfig | AirLoopConfig) -> dict[str, Any]:
    """Serialize config to a flat dict for YAML output."""
    from dataclasses import asdict

    d = asdict(cfg)
    if isinstance(cfg, UnitaryHvacConfig):
        d["type"] = "unitary_hvac"
    else:
        d["type"] = "air_loop"
    return d


def _make_objective(
    building_type: str,
    building_id: str,
    task: str,
    run_period: Literal["full_year", "winter", "summer"],
) -> optuna.Trial:
    is_vav = building_type in VAV_BUILDING_TYPES

    def objective(trial: optuna.Trial) -> float:
        if is_vav:
            cfg = _suggest_air_loop(trial)
            policy = AirLoopPolicy(cfg)
        else:
            cfg = _suggest_unitary_hvac(trial)
            policy = UnitaryHvacPolicy(cfg)

        env = b2b.new_make_env(
            building_type,
            building_id=building_id,
            task=task,
            run_period=run_period,
        )
        try:
            policy.bind_env(env)
            result = run_episode(env, policy)
            return result.total_reward
        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return float("-inf")
        finally:
            env.close()

    return objective


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    building_type: str = cfg.building_type
    climate_zone: int = int(cfg.climate_zone)
    n_trials: int = int(cfg.get("n_trials", 200))
    n_startup: int = int(cfg.get("n_startup_trials", 20))
    timeout: int | None = cfg.get("timeout_seconds")
    task: str = cfg.get("reward", {}).get("task_name", "task1")
    run_period_raw = str(cfg.get("run_period", "full_year"))
    allowed_run_periods = {"full_year", "winter", "summer"}
    if run_period_raw not in allowed_run_periods:
        raise ValueError(
            f"Invalid run_period '{run_period_raw}'. "
            f"Expected one of {sorted(allowed_run_periods)}."
        )
    run_period: Literal["full_year", "winter", "summer"] = run_period_raw  # type: ignore[assignment]
    output_dir = Path(str(cfg.get("output_dir", "configs/tuned_controllers")))

    from baselines.run_rule_based import _get_climate_zone

    building_ids = b2b.list_buildings(building_type, split="train")
    if not building_ids:
        logger.error("No buildings found for %s", building_type)
        return

    building_id = building_ids[0]
    for bid in building_ids:
        cz = _get_climate_zone(building_type, bid)
        if cz == climate_zone:
            building_id = bid
            break

    logger.info(
        "Tuning %s controller for %s (cz=%d, building=%s, task=%s, run_period=%s)",
        "air_loop" if building_type in VAV_BUILDING_TYPES else "unitary_hvac",
        building_type,
        climate_zone,
        building_id,
        task,
        run_period,
    )

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup, seed=42
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"tune_{building_type.lower()}_cz{climate_zone}",
    )

    objective = _make_objective(building_type, building_id, task, run_period)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(
        "Best trial: #%d  value=%.1f",
        study.best_trial.number,
        study.best_trial.value,
    )

    is_vav = building_type in VAV_BUILDING_TYPES
    if is_vav:
        best_cfg = AirLoopConfig(**{
            k: v
            for k, v in study.best_params.items()
            if k != "sat_aware_flow"
        }, sat_aware_flow=study.best_params.get("sat_aware_flow", True))
    else:
        best_cfg = UnitaryHvacConfig(**study.best_params)

    output_dir.mkdir(parents=True, exist_ok=True)
    if is_vav:
        fname = f"air_loop_{building_type.lower()}_cz{climate_zone}.yaml"
    else:
        fname = f"unitary_hvac_{building_type.lower()}_cz{climate_zone}.yaml"

    out_path = output_dir / fname
    cfg_dict = _config_to_dict(best_cfg)
    out_path.write_text(yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False))
    logger.info("Saved tuned config to %s", out_path)


if __name__ == "__main__":
    main()
