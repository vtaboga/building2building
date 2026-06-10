#!/usr/bin/env python3
"""Tune reactive controller parameters with Optuna.

Supports both unitary HVAC (single-zone packaged systems) and
air-loop (VAV multi-zone systems) controllers. For each trial, the
controller is evaluated on multiple buildings and the *worst* (minimum)
episode reward across them is used as the Optuna objective.

Usage with Hydra::

    python -m baselines.tune_controller experiment=tune_controller \
        building_type=OfficeSmall climate_zone=1

    python -m baselines.tune_controller experiment=tune_controller \
        building_type=OfficeMedium climate_zone=3 n_trials=100
"""

from __future__ import annotations

import gc
import logging
import random
import tempfile
from concurrent.futures import (
    CancelledError,
    Executor,
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any, Callable, Literal

import hydra
import numpy as np
import optuna
import yaml
from omegaconf import DictConfig, OmegaConf

import building2building as b2b
from baselines.controllers.air_loop import (
    AirLoopConfig,
    AirLoopPolicy,
)
from baselines.controllers.unitary_hvac import UnitaryHvacConfig, UnitaryHvacPolicy
from baselines.utils.evaluation import run_episode_reward_only

logger = logging.getLogger(__name__)

VAV_BUILDING_TYPES = {"OfficeMedium"}


def _suggest_unitary_hvac(trial: optuna.Trial) -> UnitaryHvacConfig:
    return UnitaryHvacConfig(
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
        fan_error_mode=trial.suggest_categorical(
            "fan_error_mode", ["nearest_setpoint", "center_of_band"]
        ),
    )


def _suggest_air_loop(trial: optuna.Trial) -> AirLoopConfig:
    return AirLoopConfig(
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


def _aggregate_rewards(rewards: list[float], method: str, percentile_q: float) -> float:
    """Aggregate per-building rewards into a single objective value.

    ``method="percentile"`` (default) returns a low-quantile (e.g. 25th
    percentile) — a robust-but-not-brittle target that the tuner can actually
    improve on, unlike ``min`` which is dominated by whichever building is
    unlucky on a given trial.
    """
    if not rewards:
        return float("-inf")
    if method == "min":
        return float(min(rewards))
    if method == "mean":
        return float(np.mean(rewards))
    if method == "percentile":
        return float(np.percentile(rewards, percentile_q))
    raise ValueError(
        f"Unknown aggregation method '{method}'. "
        "Expected one of: 'min', 'mean', 'percentile'."
    )


def _run_one_building(
    building_type: str,
    building_id: str,
    task: str,
    run_period: Literal["full_year", "winter", "summer"],
    cfg: UnitaryHvacConfig | AirLoopConfig,
    is_vav: bool,
) -> float:
    """Run one full episode for *building_id* and return the total reward.

    This function is defined at module top level (and therefore picklable)
    so it can be dispatched to a :class:`ProcessPoolExecutor`.  Each worker
    process creates its own EnergyPlus output directory and tears the
    simulation down aggressively before returning.

    Heavy imports (``building2building``, ``pyenergyplus``) happen inside the
    function body so that the workers — which share a ``spawn``'d interpreter
    — only pay the import cost on the first call and reuse cached modules
    across subsequent trials.
    """
    # Lazy imports inside the worker: safe under ``spawn`` start method.
    from building2building.env import setup_energyplus_path

    setup_energyplus_path()

    import building2building as worker_b2b  # noqa: N813  (alias for clarity)
    from baselines.controllers.air_loop import AirLoopPolicy as _AirLoopPolicy
    from baselines.controllers.unitary_hvac import (
        UnitaryHvacPolicy as _UnitaryHvacPolicy,
    )
    from baselines.utils.evaluation import (
        run_episode_reward_only as _run_episode_reward_only,
    )

    policy = _AirLoopPolicy(cfg) if is_vav else _UnitaryHvacPolicy(cfg)
    env = worker_b2b.make_env(
        building_type,
        building_id=building_id,
        task=task,
        run_period=run_period,
        eplus_output_dir=Path(tempfile.mkdtemp(prefix=f"b2b_tune_{building_id}_")),
    )
    try:
        policy.bind_env(env)
        return float(_run_episode_reward_only(env, policy))
    finally:
        env.close()


def _make_objective(
    building_type: str,
    building_ids: list[str],
    task: str,
    run_period: Literal["full_year", "winter", "summer"],
    aggregation: str,
    percentile_q: float,
    executor: Executor | None = None,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective that evaluates across *building_ids*.

    The score returned is the aggregated episode reward across all buildings
    (by default the 25th percentile), so that the tuner optimises for robust
    but learnable performance.  Using the strict minimum is very noisy: a
    single unlucky building can dominate the score and mask real progress.

    Each building episode is cleaned up eagerly via ``env.close()``:
    the EnergyPlus thread is joined, the native state is released, and
    the output directory is removed so EnergyPlus artefacts do not
    accumulate in ``$TMPDIR``.

    If *executor* is provided the per-building simulations are dispatched
    to it concurrently (one future per building).  Otherwise the loop is
    sequential and behaves exactly like the pre-parallel version.
    """
    is_vav = building_type in VAV_BUILDING_TYPES

    def objective(trial: optuna.Trial) -> float:
        if is_vav:
            cfg: UnitaryHvacConfig | AirLoopConfig = _suggest_air_loop(trial)
        else:
            cfg = _suggest_unitary_hvac(trial)

        if executor is None:
            return _evaluate_sequential(
                trial,
                cfg,
                is_vav,
                building_type,
                building_ids,
                task,
                run_period,
                aggregation,
                percentile_q,
            )
        return _evaluate_parallel(
            trial,
            cfg,
            is_vav,
            building_type,
            building_ids,
            task,
            run_period,
            aggregation,
            percentile_q,
            executor,
        )

    return objective


def _evaluate_sequential(
    trial: optuna.Trial,
    cfg: UnitaryHvacConfig | AirLoopConfig,
    is_vav: bool,
    building_type: str,
    building_ids: list[str],
    task: str,
    run_period: Literal["full_year", "winter", "summer"],
    aggregation: str,
    percentile_q: float,
) -> float:
    policy = AirLoopPolicy(cfg) if is_vav else UnitaryHvacPolicy(cfg)
    rewards: list[float] = []
    for bid in building_ids:
        env = b2b.make_env(
            building_type,
            building_id=bid,
            task=task,
            run_period=run_period,
            eplus_output_dir=Path(tempfile.mkdtemp(prefix=f"b2b_tune_{bid}_")),
        )
        try:
            policy.bind_env(env)
            rewards.append(run_episode_reward_only(env, policy))
        except Exception as e:
            logger.warning("Trial %d failed on %s: %s", trial.number, bid, e)
            return float("-inf")
        finally:
            env.close()
            del env
    return _aggregate_rewards(rewards, aggregation, percentile_q)


def _evaluate_parallel(
    trial: optuna.Trial,
    cfg: UnitaryHvacConfig | AirLoopConfig,
    is_vav: bool,
    building_type: str,
    building_ids: list[str],
    task: str,
    run_period: Literal["full_year", "winter", "summer"],
    aggregation: str,
    percentile_q: float,
    executor: Executor,
) -> float:
    futures = {
        executor.submit(
            _run_one_building,
            building_type,
            bid,
            task,
            run_period,
            cfg,
            is_vav,
        ): bid
        for bid in building_ids
    }
    rewards: list[float] = []
    failed = False
    try:
        for fut in as_completed(futures):
            bid = futures[fut]
            try:
                rewards.append(float(fut.result()))
            except CancelledError:
                continue
            except Exception as e:
                logger.warning("Trial %d failed on %s: %s", trial.number, bid, e)
                failed = True
                break
    finally:
        # If a worker raised we cancel the siblings so that we don't pay
        # for their remaining compute — futures already running cannot be
        # interrupted but pending ones will not start.
        if failed:
            for fut in futures:
                fut.cancel()
    if failed:
        return float("-inf")
    return _aggregate_rewards(rewards, aggregation, percentile_q)


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
    task: str = cfg.get("reward", {}).get("task_name", "task_occ_emed")
    run_period_raw = str(cfg.get("run_period", "full_year"))
    allowed_run_periods = {"full_year", "winter", "summer"}
    if run_period_raw not in allowed_run_periods:
        raise ValueError(
            f"Invalid run_period '{run_period_raw}'. "
            f"Expected one of {sorted(allowed_run_periods)}."
        )
    run_period: Literal["full_year", "winter", "summer"] = run_period_raw  # type: ignore[assignment]
    output_dir = Path(str(cfg.get("output_dir", "configs/tuned_controllers")))

    n_eval_buildings: int = int(cfg.get("n_eval_buildings", 5))

    aggregation: str = str(cfg.get("aggregation", "percentile"))
    allowed_aggregations = {"min", "mean", "percentile"}
    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. "
            f"Expected one of {sorted(allowed_aggregations)}."
        )
    percentile_q: float = float(cfg.get("percentile_q", 25.0))
    if not 0.0 < percentile_q < 100.0:
        raise ValueError(f"percentile_q must be in (0, 100); got {percentile_q}.")
    storage_dir_cfg = cfg.get("storage_dir", None)
    storage_dir = (
        Path(str(storage_dir_cfg)) if storage_dir_cfg is not None else output_dir
    )
    n_building_workers: int = int(cfg.get("n_building_workers", 1))
    if n_building_workers < 1:
        raise ValueError(f"n_building_workers must be >= 1; got {n_building_workers}.")

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=42)
    study_name = f"tune_{building_type.lower()}_cz{climate_zone}_{task}"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    logger.info(
        "Study storage: %s (%d completed trials already in DB)",
        storage_url,
        completed,
    )
    remaining_trials = max(0, n_trials - completed)

    # ── Fixed eval-building set ──────────────────────────────────────────
    # The evaluation buildings are drawn ONCE per study (seeded) and then
    # persisted in the study's user attributes.  Every resumed job – and any
    # later analysis script that inspects the study – therefore sees the
    # exact same pool of buildings, regardless of changes to the upstream
    # building registry or the ``n_eval_buildings`` config value.
    stored_eval_ids = study.user_attrs.get("eval_building_ids")
    if stored_eval_ids is not None:
        eval_ids = [str(b) for b in stored_eval_ids]
        logger.info(
            "Re-using %d eval buildings stored in study user_attrs",
            len(eval_ids),
        )
    else:
        if building_type in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
            # SingleFamilyHouse has no CZ mapping — use all test buildings.
            matching_ids = b2b.list_buildings(building_type, split="test")
        else:
            matching_ids = b2b.list_buildings_by_climate_zone(
                building_type, climate_zone, split="test"
            )

        if not matching_ids:
            logger.error(
                "No test buildings found for %s / cz=%s",
                building_type,
                climate_zone,
            )
            return

        rng = random.Random(42)
        if len(matching_ids) <= n_eval_buildings:
            eval_ids = list(matching_ids)
        else:
            eval_ids = rng.sample(matching_ids, n_eval_buildings)

        study.set_user_attr("eval_building_ids", eval_ids)
        study.set_user_attr(
            "eval_building_selection",
            {
                "seed": 42,
                "n_eval_buildings": n_eval_buildings,
                "pool_size": len(matching_ids),
                "building_type": building_type,
                "climate_zone": climate_zone,
            },
        )
        logger.info(
            "Drew %d eval buildings (seed=42, pool=%d) and stored them "
            "in study user_attrs",
            len(eval_ids),
            len(matching_ids),
        )

    logger.info(
        "Tuning %s controller for %s (cz=%d, %d eval buildings, task=%s, "
        "run_period=%s)",
        "air_loop" if building_type in VAV_BUILDING_TYPES else "unitary_hvac",
        building_type,
        climate_zone,
        len(eval_ids),
        task,
        run_period,
    )
    logger.info("Eval building IDs: %s", eval_ids)

    # ── W&B logging ──────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(OmegaConf.select(wandb_cfg, "enabled", default=False))
    callbacks: list[Any] = []

    if use_wandb:
        try:
            import wandb
            from optuna.integration.wandb import WeightsAndBiasesCallback

            wandb.init(
                project=OmegaConf.select(wandb_cfg, "project", default="b2b-baselines"),
                entity=OmegaConf.select(wandb_cfg, "entity", default=None),
                tags=list(OmegaConf.select(wandb_cfg, "tags", default=[])),
                name=f"tune_{building_type}_cz{climate_zone}_{task}",
                group=f"tune_controller_{building_type}",
                config={
                    "building_type": building_type,
                    "climate_zone": climate_zone,
                    "task": task,
                    "n_trials": n_trials,
                    "n_eval_buildings": len(eval_ids),
                    "eval_building_ids": eval_ids,
                    "run_period": run_period,
                },
            )
            wandb_callback = WeightsAndBiasesCallback(
                metric_name="worst_reward",
            )
            callbacks.append(wandb_callback)
            logger.info(
                "W&B logging enabled (project=%s)",
                OmegaConf.select(wandb_cfg, "project", default="b2b-baselines"),
            )
        except ImportError:
            logger.warning("wandb or optuna[wandb] not installed; skipping W&B logging")
    else:
        logger.info("W&B logging disabled (wandb.enabled=false)")

    _best_so_far = float("-inf")

    def _track_best(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal _best_so_far
        if trial.value is not None and trial.value > _best_so_far:
            _best_so_far = trial.value
        if use_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "best_reward_so_far": _best_so_far,
                            "trial_number": trial.number,
                        }
                    )
            except Exception:
                pass

    callbacks.append(_track_best)

    def _gc_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        gc.collect()

    callbacks.append(_gc_callback)

    # ── Within-trial parallelism ─────────────────────────────────────────
    # When ``n_building_workers > 1`` the per-building EnergyPlus runs of
    # each trial are dispatched to a persistent ``ProcessPoolExecutor``
    # using the ``spawn`` start method (EnergyPlus is not fork-safe).  The
    # executor is reused across trials so the b2b / pyenergyplus import
    # cost is paid only on the first trial.  Each worker uses ~1 CPU core
    # and several hundred MB of RAM – bump ``--cpus-per-task`` and
    # ``--mem`` in the SLURM script accordingly.
    effective_workers = min(n_building_workers, len(eval_ids))
    executor: ProcessPoolExecutor | None = None
    if effective_workers > 1:
        import multiprocessing as _mp

        ctx = _mp.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx)
        logger.info(
            "Within-trial parallelism: %d worker processes (spawn)",
            effective_workers,
        )
    else:
        logger.info(
            "Within-trial parallelism: disabled (n_building_workers=%d, "
            "eval_ids=%d)",
            n_building_workers,
            len(eval_ids),
        )

    objective = _make_objective(
        building_type,
        eval_ids,
        task,
        run_period,
        aggregation=aggregation,
        percentile_q=percentile_q,
        executor=executor,
    )
    logger.info(
        "Objective aggregation: %s%s",
        aggregation,
        f" (q={percentile_q:.1f})" if aggregation == "percentile" else "",
    )
    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=timeout,
            callbacks=callbacks,
            catch=(ValueError,),
            gc_after_trial=True,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if use_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.summary["best_trial_number"] = study.best_trial.number
                wandb.run.summary["best_reward"] = study.best_trial.value
                wandb.run.summary["best_params"] = study.best_params
                wandb.finish()
        except Exception:
            pass

    logger.info(
        "Best trial: #%d  value=%.1f",
        study.best_trial.number,
        study.best_trial.value,
    )

    is_vav = building_type in VAV_BUILDING_TYPES
    if is_vav:
        best_cfg = AirLoopConfig(
            **{k: v for k, v in study.best_params.items() if k != "sat_aware_flow"},
            sat_aware_flow=study.best_params.get("sat_aware_flow", True),
        )
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
