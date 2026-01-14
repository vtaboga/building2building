from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algorithms.baselines.common import (
    RolloutPaths,
    find_controlled_zone_air_temp_index,
    make_rollout_paths,
    require_env_metadata_list_str,
)
from algorithms.baselines.controllers.fan_coil_constant import compute_fan_command_constant
from algorithms.baselines.controllers.unitary_pi import PIState, compute_fan_command_pi
from algorithms.baselines.plotting import plot_actuators_dual_axis
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
        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")

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

        # Node setpoints: keep them at "classic" values (configurable) while airflow does the work.
        t_heat = float(getattr(cfg.policy, "heating_coil_setpoint_c", 35.0))
        t_supp = float(getattr(cfg.policy, "supplemental_coil_setpoint_c", 45.0))
        t_cool = float(getattr(cfg.policy, "cooling_coil_setpoint_c", 12.0))
        # Supply/outlet node setpoint (unitary system air outlet node).
        t_outlet = float(getattr(cfg.policy, "outlet_node_setpoint_c", 22.0))

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

        all_rows: list[dict[str, float]] = []

        for ep in range(n_episodes):
            obs, _info = env.reset()
            done = False
            step = 0
            pi_state = PIState(integral=0.0)

            while not done and step < max_steps:
                tz = float(np.asarray(obs, dtype=float).reshape(-1)[temp_idx])

                # Allow runtime override (Hydra sweeps) without reloading config objects.
                target = float(getattr(cfg.policy, "target_temp_c", target))
                deadband = float(getattr(cfg.policy, "deadband_c", deadband))

                action_cmd = np.zeros((len(act_names),), dtype=float)

                for idx in idxs.idx_avail:
                    action_cmd[int(idx)] = availability_on

                # Fan command (either constant or PI-controlled).
                if policy_type == "unitary_pi":
                    fan_cmd = compute_fan_command_pi(
                        state=pi_state,
                        tz_c=tz,
                        target_c=target,
                        deadband_c=deadband,
                        fan_base_kg_s=fan_base,
                        kp=kp,
                        ki=ki,
                        integral_limit=integral_limit,
                    )
                else:
                    fan_cmd = compute_fan_command_constant(
                        fan_mass_flow_kg_s=fan_mdot_const
                    )

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

                for idx in idxs.idx_heat_nodes:
                    action_cmd[int(idx)] = t_heat
                for idx in idxs.idx_supp_nodes:
                    action_cmd[int(idx)] = t_supp
                for idx in idxs.idx_cool_nodes:
                    action_cmd[int(idx)] = t_cool
                for idx in idxs.idx_outlet_nodes:
                    action_cmd[int(idx)] = t_outlet

                obs2, reward, terminated, truncated, _info = env.step(action_cmd)

                row: dict[str, float] = {
                    "episode": float(ep),
                    "step": float(step),
                    "reward": float(reward),
                }

                obs_arr = np.asarray(obs2, dtype=float).reshape(-1)
                for i, name in enumerate(obs_names):
                    if i < len(obs_arr):
                        row[f"obs::{name}"] = float(obs_arr[i])

                act_arr = np.asarray(action_cmd, dtype=float).reshape(-1)
                for i, name in enumerate(act_names):
                    if i < len(act_arr):
                        row[f"act::{name}"] = float(act_arr[i])
                all_rows.append(row)

                obs = obs2
                done = bool(terminated or truncated)
                step += 1

            logger.info(f"episode={ep} steps={step} saved_rows={len(all_rows)}")

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

        _get_plot_timeseries()(
            df=df,
            x="step",
            y_cols=temp_cols[:25],
            out_path=paths.plot_temperature_path,
            title=f"Zone Temperatures (target={target:.1f}C deadband=±{deadband:.1f}C)",
            ylabel="Temperature [C]",
            hlines=[
                (target, "target"),
                (target - deadband, "target-deadband"),
                (target + deadband, "target+deadband"),
            ],
        )

        energy_cols = [c for c in obs_cols if c.lower().endswith("energy_electricity")] + [
            c for c in obs_cols if c.lower().endswith("energy_gas")
        ]
        _get_plot_timeseries()(
            df=df,
            x="step",
            y_cols=energy_cols,
            out_path=paths.plot_energy_path,
            title="HVAC Energy (per-area, per-timestep)",
            ylabel="Wh / m2 / timestep",
        )

        plot_actuators_dual_axis(
            df=df,
            x="step",
            act_cols=act_cols,
            out_path=paths.plot_actuators_path,
        )

        logger.info(f"Saved raw rollout CSV: {paths.csv_path}")
        logger.info(f"Saved raw rollout NPZ: {paths.npz_path}")
        logger.info(f"Saved plots under: {paths.out_dir}")
        return paths
    finally:
        try:
            env.close()
        except Exception:
            pass

