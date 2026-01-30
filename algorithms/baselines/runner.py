from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algorithms.wandb_utils import (
    finish_wandb_if_started,
    init_wandb_from_config,
)
from algorithms.baselines.common import (
    RolloutPaths,
    find_day_of_week_index,
    find_controlled_zone_air_temp_index,
    find_time_of_day_index,
    make_rollout_paths,
    require_env_metadata_list_str,
)
from algorithms.baselines.controllers.fan_coil_constant import compute_fan_command_constant
from algorithms.baselines.controllers.unitary_airflow_first_sat import (
    AirflowFirstSatState,
    compute_outlet_sat_sp_airflow_first,
)
from algorithms.baselines.controllers.unitary_pi import (
    PIState,
    UnitaryPIMode,
    compute_fan_command_pi,
)
from algorithms.baselines.unitary_actuators import select_unitary_actuator_indices

logger = logging.getLogger(__name__)


def _get_make_env():
    """
    Resolve `make_env` dynamically from the `algorithms.baselines` package.

    This preserves the existing test pattern where tests monkeypatch
    `algorithms.baselines.make_env` and expect it to affect `run_baseline_rollout`.
    """
    import algorithms.baselines as baselines_pkg

    return baselines_pkg.make_env


def _get_plot_timeseries():
    import algorithms.baselines as baselines_pkg

    return baselines_pkg.plot_timeseries


def run_baseline_rollout(cfg: DictConfig, *, run_dir: Path | None = None) -> RolloutPaths:
    """
    Rollout a simulation with a baseline policy.

    Expected to be launched under Hydra, so by default `run_dir` is `Path.cwd()`.
    """
    if run_dir is None:
        if HydraConfig.initialized():
            run_dir = Path(HydraConfig.get().runtime.output_dir)
        else:
            run_dir = Path.cwd()
    paths = make_rollout_paths(run_dir)

    with paths.config_path.open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    eplus_output_dir = run_dir / "eplus_outputs"
    env = _get_make_env()(config=cfg, eplus_output_dir=str(eplus_output_dir))
    try:
        # Log which *actual* building was fetched/selected (not just the query config).
        building_source_meta: dict[str, object] = {}
        if hasattr(env, "metadata") and isinstance(env.metadata, dict):
            raw = env.metadata.get("building_source_metadata")
            if isinstance(raw, dict):
                building_source_meta = dict(raw)

        if building_source_meta:
            # Write to disk for reproducibility / debugging.
            try:
                with (paths.out_dir / "building_info.json").open("w", encoding="utf-8") as f:
                    json.dump(building_source_meta, f, indent=2)
            except Exception as e:
                logger.warning("Failed to write building_info.json: %s", e)

        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")

        # Start W&B early so we can log scalar time series (no tables / no images).
        wandb_run, started_here = init_wandb_from_config(cfg, run_dir=run_dir)

        # Print action bounds for debugging.
        try:
            if (
                hasattr(env, "action_space")
                and hasattr(env.action_space, "low")
                and hasattr(env.action_space, "high")
            ):
                lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
                highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
                print("\n=== Action bounds (low/high) ===", flush=True)
                for i, name in enumerate(act_names):
                    if i < len(lows) and i < len(highs):
                        print(
                            f"{i:3d} {name}: [{lows[i]:.3f}, {highs[i]:.3f}]",
                            flush=True,
                        )
                print("=== End action bounds ===\n", flush=True)
        except Exception as e:
            logger.warning(f"Could not print action bounds: {e}")

        # We still pick a representative controlled-zone temperature for plotting/debugging.
        controlled_zones = None
        try:
            controlled_zones = require_env_metadata_list_str(env, "controlled_zones")
        except Exception:
            controlled_zones = None
        temp_idx = find_controlled_zone_air_temp_index(
            obs_names, controlled_zones=controlled_zones
        )

        idxs = select_unitary_actuator_indices(act_names)

        max_steps = int(getattr(cfg.env, "max_steps"))
        n_episodes = int(getattr(cfg, "n_episodes"))

        policy_type = str(getattr(cfg.policy, "type", "fan_coil_constant")).strip()

        # Target temperature used for PI control (and for plot annotation).
        target = float(getattr(cfg.policy, "target_temp_c", 21.0))
        deadband = float(getattr(cfg.policy, "deadband_c", 1.0))

        # Optional time-based target setpoint schedule.
        #
        # When enabled, we compute the active target setpoint from:
        # - time_of_day (hours)
        # - day_of_week (EnergyPlus convention: 1=Sunday, ..., 7=Saturday)
        schedule_cfg = getattr(cfg.policy, "target_schedule", None)
        schedule_enabled = bool(getattr(schedule_cfg, "enabled", False)) if schedule_cfg else False
        time_of_day_idx: int | None = None
        day_of_week_idx: int | None = None
        weekend_days: set[int] = {1, 7}
        weekend_target_c = 21.0
        weekday_target_c = 21.0
        weekday_setback_target_c = 18.0
        weekday_setback_start_hour = 9.0
        weekday_setback_end_hour = 16.0

        if schedule_enabled:
            time_of_day_idx = find_time_of_day_index(obs_names)
            day_of_week_idx = find_day_of_week_index(obs_names)

            raw_weekend_days = getattr(schedule_cfg, "weekend_days", [1, 7])
            try:
                weekend_days_list = list(raw_weekend_days)
            except Exception as e:
                raise RuntimeError(
                    "policy.target_schedule.weekend_days must be a list-like of ints"
                ) from e
            if not weekend_days_list or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool)
                for x in weekend_days_list
            ):
                raise RuntimeError(
                    "policy.target_schedule.weekend_days must be a list-like of ints"
                )
            weekend_days = {int(x) for x in weekend_days_list}

            weekend_target_c = float(getattr(schedule_cfg, "weekend_target_c"))
            weekday_target_c = float(getattr(schedule_cfg, "weekday_target_c"))
            weekday_setback_target_c = float(
                getattr(schedule_cfg, "weekday_setback_target_c")
            )
            weekday_setback_start_hour = float(
                getattr(schedule_cfg, "weekday_setback_start_hour")
            )
            weekday_setback_end_hour = float(
                getattr(schedule_cfg, "weekday_setback_end_hour")
            )

        def _scheduled_target_c(*, obs_arr: np.ndarray) -> float:
            if not schedule_enabled:
                return float(target)
            assert time_of_day_idx is not None and day_of_week_idx is not None
            tod = float(obs_arr[time_of_day_idx])
            # Normalize to [0, 24) for robust comparisons even if E+ returns 24/25 at boundaries.
            hour = float(tod % 24.0)

            dow = int(round(float(obs_arr[day_of_week_idx])))
            if dow in weekend_days:
                return float(weekend_target_c)

            # Weekday.
            if weekday_setback_start_hour <= hour < weekday_setback_end_hour:
                return float(weekday_setback_target_c)
            return float(weekday_target_c)

        # Unitary PI baseline: only control fan airflow + a single outlet node temperature setpoint.
        #
        # Prefer explicit outlet temps if provided; otherwise fall back to legacy keys
        # (previously used to drive multiple node setpoints).
        outlet_heat_c = float(
            getattr(
                cfg.policy,
                "outlet_temp_heating_c",
                getattr(cfg.policy, "heating_coil_setpoint_c", 35.0),
            )
        )
        outlet_cool_c = float(
            getattr(
                cfg.policy,
                "outlet_temp_cooling_c",
                getattr(cfg.policy, "cooling_coil_setpoint_c", 12.0),
            )
        )

        # Force HVAC system available while commanding airflow + setpoints.
        availability_on = float(getattr(cfg.policy, "availability_on", 2.0))

        # Fan airflow control configuration
        fan_mdot_const = float(getattr(cfg.policy, "fan_mass_flow_kg_s", 1.0))
        kp = float(getattr(cfg.policy, "kp", 0.0))
        ki = float(getattr(cfg.policy, "ki", 0.0))
        integral_limit = float(getattr(cfg.policy, "integral_limit", 1e6))
        fan_base = float(getattr(cfg.policy, "fan_base_kg_s", fan_mdot_const))

        # Optional overrides. If absent, we clamp using the action space bounds.
        fan_min_cfg = getattr(cfg.policy, "fan_min_kg_s", None)
        fan_max_cfg = getattr(cfg.policy, "fan_max_kg_s", None)

        # Option 1: airflow-first with SAT trim/reset
        sat_min_c = float(getattr(cfg.policy, "sat_min_c", 10.0))
        sat_max_c = float(getattr(cfg.policy, "sat_max_c", 45.0))
        sat_step_c = float(getattr(cfg.policy, "sat_step_c", 0.5))
        sat_rate_limit_c_per_step = float(
            getattr(cfg.policy, "sat_rate_limit_c_per_step", 1e6)
        )
        sat_saturation_steps = int(getattr(cfg.policy, "sat_saturation_steps", 1))

        all_rows: list[dict[str, float]] = []
        episode_returns: list[float] = []

        for ep in range(n_episodes):
            obs, _info = env.reset()
            done = False
            step = 0
            ep_return = 0.0
            pi_state = PIState(integral=0.0)
            last_time_of_day: float | None = None
            sat_state = AirflowFirstSatState(sat_sp_c=float(target))

            while not done and step < max_steps:
                tz = float(np.asarray(obs, dtype=float).reshape(-1)[temp_idx])

                obs_arr_now = np.asarray(obs, dtype=float).reshape(-1)

                # Allow runtime override (Hydra sweeps) without reloading config objects.
                target = float(getattr(cfg.policy, "target_temp_c", target))
                deadband = float(getattr(cfg.policy, "deadband_c", deadband))
                target_now = float(_scheduled_target_c(obs_arr=obs_arr_now))

                action_cmd = np.zeros((len(act_names),), dtype=float)

                for idx in idxs.idx_avail:
                    action_cmd[int(idx)] = availability_on

                # Fan command (either constant or PI-controlled).
                if policy_type in ("unitary_pi", "unitary_airflow_first_sat"):
                    # Determine operating mode from the zone temperature.
                    if tz < target_now - deadband:
                        mode: UnitaryPIMode = "heating"
                    elif tz > target_now + deadband:
                        mode = "cooling"
                    else:
                        mode = "deadband"

                    fan_cmd = compute_fan_command_pi(
                        state=pi_state,
                        tz_c=tz,
                        target_c=target_now,
                        deadband_c=deadband,
                        fan_base_kg_s=fan_base,
                        kp=kp,
                        ki=ki,
                        integral_limit=integral_limit,
                        mode=mode,
                    )
                else:
                    fan_cmd = compute_fan_command_constant(
                        fan_mass_flow_kg_s=fan_mdot_const
                    )

                # Compute fan command bounds (used by option 1 SAT logic).
                # Prefer action space bounds when present; fall back to config or a wide range.
                fan_low = float("-inf")
                fan_high = float("inf")
                if idxs.idx_fans and hasattr(env, "action_space"):
                    try:
                        lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
                        highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
                        i0 = int(idxs.idx_fans[0])
                        if 0 <= i0 < len(lows) and 0 <= i0 < len(highs):
                            fan_low = float(lows[i0])
                            fan_high = float(highs[i0])
                    except Exception:
                        pass
                if fan_min_cfg is not None:
                    fan_low = max(fan_low, float(fan_min_cfg))
                if fan_max_cfg is not None:
                    fan_high = min(fan_high, float(fan_max_cfg))

                fan_cmd_for_logic = float(np.clip(float(fan_cmd), fan_low, fan_high))

                for idx in idxs.idx_fans:
                    i = int(idx)
                    cmd_i = float(fan_cmd)

                    # Clamp using action space bounds when available (preferred).
                    if (
                        hasattr(env, "action_space")
                        and hasattr(env.action_space, "low")
                        and hasattr(env.action_space, "high")
                    ):
                        lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
                        highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
                        if i < len(lows) and i < len(highs):
                            cmd_i = float(
                                np.clip(cmd_i, float(lows[i]), float(highs[i]))
                            )

                    # Optional config-level clamp (applied after action-space clamp).
                    if fan_min_cfg is not None:
                        cmd_i = max(cmd_i, float(fan_min_cfg))
                    if fan_max_cfg is not None:
                        cmd_i = min(cmd_i, float(fan_max_cfg))

                    action_cmd[i] = cmd_i

                # Node setpoint management:
                # For `unitary_pi`, we **only** set a fixed outlet-node setpoint
                # (mode-dependent) and do not override other node setpoints.
                if policy_type == "unitary_pi":
                    if mode == "heating":
                        outlet_sp = outlet_heat_c
                    elif mode == "cooling":
                        outlet_sp = outlet_cool_c
                    else:
                        # In deadband we should not force heating or cooling; using
                        # the comfort target lets the HVAC float naturally.
                        outlet_sp = target_now
                    for idx in idxs.idx_outlet_nodes:
                        action_cmd[int(idx)] = float(outlet_sp)
                elif policy_type == "unitary_airflow_first_sat":
                    # Option 1: airflow-first, then SAT trim/reset at airflow limits.
                    outlet_sp = compute_outlet_sat_sp_airflow_first(
                        state=sat_state,
                        tz_c=tz,
                        target_c=target_now,
                        deadband_c=deadband,
                        fan_cmd_kg_s=fan_cmd_for_logic,
                        fan_min_kg_s=float(fan_low),
                        fan_max_kg_s=float(fan_high),
                        mode=mode,
                        outlet_sp_heating_c=outlet_heat_c,
                        outlet_sp_cooling_c=outlet_cool_c,
                        sat_min_c=sat_min_c,
                        sat_max_c=sat_max_c,
                        sat_step_c=sat_step_c,
                        sat_rate_limit_c_per_step=sat_rate_limit_c_per_step,
                        sat_saturation_steps=sat_saturation_steps,
                    )
                    for idx in idxs.idx_outlet_nodes:
                        i = int(idx)
                        cmd_i = float(outlet_sp)
                        # Clamp using action space bounds when available (preferred).
                        if (
                            hasattr(env, "action_space")
                            and hasattr(env.action_space, "low")
                            and hasattr(env.action_space, "high")
                        ):
                            lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
                            highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
                            if i < len(lows) and i < len(highs):
                                cmd_i = float(
                                    np.clip(cmd_i, float(lows[i]), float(highs[i]))
                                )
                        action_cmd[i] = cmd_i

                obs2, reward, terminated, truncated, _info = env.step(action_cmd)
                ep_return += float(reward)

                obs_arr = np.asarray(obs2, dtype=float).reshape(-1)
                conditioned_tz = (
                    float(obs_arr[int(temp_idx)]) if int(temp_idx) < len(obs_arr) else float("nan")
                )

                row: dict[str, float] = {
                    "episode": float(ep),
                    "step": float(step),
                    "global_step": float(int(ep) * int(max_steps) + int(step)),
                    "reward": float(reward),
                    # For plotting (setpoint vs conditioned zone temperature).
                    "target_temp_c": float(target_now),
                    "conditioned_zone_temp_c": float(conditioned_tz),
                }
                for i, name in enumerate(obs_names):
                    if i < len(obs_arr):
                        row[f"obs::{name}"] = float(obs_arr[i])

                act_arr = np.asarray(action_cmd, dtype=float).reshape(-1)
                for i, name in enumerate(act_names):
                    if i < len(act_arr):
                        row[f"act::{name}"] = float(act_arr[i])
                all_rows.append(row)

                # W&B scalar time series:
                # Keep per-step logging light (reward/actuators only). Temperature plots are
                # logged as multi-line charts from the final dataframe for readability.
                if wandb_run is not None:
                    try:
                        import wandb  # type: ignore

                        log_payload: dict[str, float] = {
                            "rollout/reward": float(reward),
                        }

                        # Also log actuator means for convenience.
                        fan_vals = [float(action_cmd[int(i)]) for i in idxs.idx_fans]
                        sp_vals = [float(action_cmd[int(i)]) for i in idxs.idx_outlet_nodes]
                        avail_vals = [float(action_cmd[int(i)]) for i in idxs.idx_avail]
                        if fan_vals:
                            log_payload["rollout/actuators/fan_mass_flow_kg_s_mean"] = float(
                                np.mean(fan_vals)
                            )
                        if sp_vals:
                            log_payload["rollout/actuators/outlet_temp_sp_c_mean"] = float(
                                np.mean(sp_vals)
                            )
                        if avail_vals:
                            log_payload["rollout/actuators/availability_mean"] = float(
                                np.mean(avail_vals)
                            )

                        # Use a monotonically increasing global step for multi-episode runs.
                        global_step = int(ep) * int(max_steps) + int(step)
                        wandb.log(log_payload, step=global_step)
                    except Exception as e:
                        logger.warning("wandb per-step logging failed: %s", e)

                obs = obs2
                done = bool(terminated or truncated)
                step += 1

            logger.info(f"episode={ep} steps={step} saved_rows={len(all_rows)}")
            episode_returns.append(float(ep_return))

        df = pd.DataFrame(all_rows)
        df.to_csv(paths.csv_path, index=False)

        obs_cols = [c for c in df.columns if c.startswith("obs::")]
        act_cols = [c for c in df.columns if c.startswith("act::")]
        np.savez_compressed(
            paths.npz_path,
            episode=df["episode"].to_numpy(dtype=np.int32),
            step=df["step"].to_numpy(dtype=np.int32),
            reward=df["reward"].to_numpy(dtype=float),
            obs_names=np.asarray(obs_cols, dtype=object),
            act_names=np.asarray(act_cols, dtype=object),
            obs=df[obs_cols].to_numpy(dtype=float)
            if obs_cols
            else np.zeros((len(df), 0)),
            act=df[act_cols].to_numpy(dtype=float)
            if act_cols
            else np.zeros((len(df), 0)),
        )

        zone_temp_cols = [c for c in obs_cols if "zone air temperature" in c.lower()]
        outdoor_temp_cols = [c for c in obs_cols if c.lower().endswith("outdoor_temperature")]
        temp_cols = zone_temp_cols + outdoor_temp_cols

        # Energy observations are per-area, per-timestep energy [Wh/m2 per timestep]
        # (see `building2building/simulator/observation_spaces.py`).
        #
        # Convert to average power per area [kW/m2] using the timestep duration derived
        # from the `time_of_day` observation.
        time_col = "obs::time_of_day"
        dt_hours: float | None = None
        if time_col in df.columns and len(df) >= 2:
            tod = pd.to_numeric(df[time_col], errors="coerce").astype(float)
            # Difference in hours, modulo 24 to handle midnight wrap-around.
            diffs = (tod.diff() % 24.0).dropna()
            diffs = diffs[(diffs > 0.0) & (diffs <= 6.0)]  # keep plausible timesteps
            if len(diffs) > 0:
                dt_hours = float(diffs.median())
        if dt_hours is None or not (dt_hours > 0.0):
            # Default pipeline timestep is 4/hour = 0.25h (15min).
            dt_hours = 0.25

        energy_cols_wh_m2 = [c for c in obs_cols if c.lower().endswith("energy_electricity")] + [
            c for c in obs_cols if c.lower().endswith("energy_gas")
        ]
        energy_cols_kw_m2: list[str] = []
        for c in energy_cols_wh_m2:
            out_c = f"{c}_kw_per_m2"
            df[out_c] = pd.to_numeric(df[c], errors="coerce").astype(float) / dt_hours / 1000.0
            energy_cols_kw_m2.append(out_c)

        try:
            if wandb_run is not None:
                # Log controller type for easy filtering in the W&B UI.
                try:
                    import wandb  # type: ignore

                    wandb.summary["baseline/controller_type"] = str(policy_type)
                    wandb.summary["baseline/policy_type"] = str(policy_type)
                    wandb.summary["rollout/timestep_hours"] = float(dt_hours)
                    wandb.summary["rollout/episode_return_mean"] = float(
                        np.mean(episode_returns) if episode_returns else 0.0
                    )

                    # Also add as a run tag (opt-in, best-effort).
                    try:
                        existing = list(getattr(wandb_run, "tags", []) or [])
                        tag = f"baseline:{policy_type}"
                        if tag not in existing:
                            existing.append(tag)
                        wandb_run.tags = existing  # type: ignore[attr-defined]
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("Failed to log baseline controller type to wandb: %s", e)

                # Attach building identification to the run summary.
                try:
                    import wandb  # type: ignore

                    # Prefer simulator-provided metadata, but fall back to config so
                    # these fields are never empty in W&B.
                    source = building_source_meta.get("source") if building_source_meta else None
                    if source is None:
                        source = OmegaConf.select(cfg, "bldg.source")
                    if source is not None:
                        wandb.summary["building/source"] = source

                    # 1) From simulator metadata (if present)
                    if building_source_meta:
                        for k in (
                            "dataset_row_index",
                            "idf_filename",
                            "schedule_filename",
                            "epw_filename",
                            "geometry_unit_type",
                            "geometry_building_num_units",
                            "year_built",
                        ):
                            v = building_source_meta.get(k)
                            if v is not None:
                                wandb.summary[f"building/{k}"] = v

                    # 2) Fallback from config (baseline configs use `bldg.bldg.*`)
                    cfg_fallbacks = {
                        "geometry_unit_type": OmegaConf.select(cfg, "bldg.bldg.geometry_unit_type"),
                        "geometry_building_num_units": OmegaConf.select(
                            cfg, "bldg.bldg.geometry_building_num_units"
                        ),
                        "year_built": OmegaConf.select(cfg, "bldg.bldg.year_built"),
                    }
                    for k, v in cfg_fallbacks.items():
                        if v is not None and wandb.summary.get(f"building/{k}") is None:
                            wandb.summary[f"building/{k}"] = v
                except Exception as e:
                    logger.warning("Failed to log building info to wandb: %s", e)

                try:
                    import wandb  # type: ignore

                    wandb.summary["rollout/mean_reward"] = float(df["reward"].mean())
                except Exception:
                    pass

                # Add two easy-to-read baseline temperature charts:
                # - Outdoor + all zone temperatures (single chart, multiple series)
                # - Setpoint + conditioned zone temperature (single chart, multiple series)
                try:
                    import wandb  # type: ignore

                    # Identify temperature columns in the raw rollout dataframe.
                    zone_temp_cols = [
                        c for c in obs_cols if "zone air temperature" in c.lower()
                    ]
                    outdoor_temp_cols = [
                        c for c in obs_cols if c.lower().endswith("outdoor_temperature")
                    ]

                    plot_cols = [
                        "global_step",
                        *outdoor_temp_cols,
                        *zone_temp_cols,
                        "target_temp_c",
                        "conditioned_zone_temp_c",
                    ]
                    plot_cols = [c for c in plot_cols if c in df.columns]

                    if "global_step" in plot_cols and len(plot_cols) > 1:
                        plot_df = df[plot_cols].copy()
                        # Ensure numeric types for W&B tables/plots.
                        for c in plot_cols:
                            plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce").astype(float)

                        # Log the underlying table too (handy for debugging / ad-hoc plots).
                        wandb.log({"rollout/tables/temperature_timeseries": wandb.Table(dataframe=plot_df)})

                        x_vals = plot_df["global_step"].tolist()

                        y_outdoor_and_zones = [
                            c
                            for c in [*outdoor_temp_cols, *zone_temp_cols]
                            if c in plot_df.columns
                        ]
                        if y_outdoor_and_zones:
                            ys = [plot_df[c].tolist() for c in y_outdoor_and_zones]
                            keys = [c.replace("obs::", "") for c in y_outdoor_and_zones]
                            wandb.log(
                                {
                                    "rollout/plots/outdoor_and_zones_temperature": wandb.plot.line_series(
                                        xs=x_vals,
                                        ys=ys,
                                        keys=keys,
                                        title="Outdoor + Zone Air Temperatures",
                                        xname="global_step",
                                    )
                                }
                            )

                        y_setpoint = [
                            c
                            for c in ("target_temp_c", "conditioned_zone_temp_c")
                            if c in plot_df.columns
                        ]
                        if len(y_setpoint) >= 2:
                            ys = [plot_df[c].tolist() for c in y_setpoint]
                            keys = ["setpoint_c", "conditioned_zone_temp_c"]
                            wandb.log(
                                {
                                    "rollout/plots/setpoint_and_conditioned_zone_temperature": wandb.plot.line_series(
                                        xs=x_vals,
                                        ys=ys,
                                        keys=keys,
                                        title="Setpoint + Conditioned Zone Temperature",
                                        xname="global_step",
                                    )
                                }
                            )
                except Exception as e:
                    logger.warning("Failed to log baseline temperature plots to wandb: %s", e)
        finally:
            finish_wandb_if_started(wandb_run, started_here=started_here)

        logger.info(f"Saved raw rollout CSV: {paths.csv_path}")
        logger.info(f"Saved raw rollout NPZ: {paths.npz_path}")
        return paths
    finally:
        try:
            env.close()
        except Exception:
            pass

