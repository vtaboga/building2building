from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from gymnasium.spaces import MultiDiscrete
from omegaconf import OmegaConf

from algorithms.baselines import (
    OnOffSensibleLoadPolicy,
    PIDSensibleLoadPolicy,
    TrimAndRespondSensibleLoadPolicy,
)
from algorithms.utils import make_env
from building2building.simulator.action_spaces import hvac_actuators_multidiscrete_transform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutPaths:
    out_dir: Path
    csv_path: Path
    npz_path: Path
    config_path: Path
    plot_temperature_path: Path
    plot_energy_path: Path
    plot_actuators_path: Path


def _make_rollout_paths(base_dir: Path) -> RolloutPaths:
    out_dir = base_dir / "ha_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    return RolloutPaths(
        out_dir=out_dir,
        csv_path=out_dir / "rollout.csv",
        npz_path=out_dir / "rollout.npz",
        config_path=out_dir / "config_resolved.json",
        plot_temperature_path=out_dir / "temperature.png",
        plot_energy_path=out_dir / "energy.png",
        plot_actuators_path=out_dir / "actuators.png",
    )


def _require_env_metadata_list_str(env, key: str) -> list[str]:
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("env.metadata missing")
    raw = env.metadata.get(key)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"env.metadata['{key}'] must be a list[str]")
    return raw


def _find_first_zone_air_temp_index(observation_names: list[str]) -> int:
    for i, name in enumerate(observation_names):
        if str(name).lower().startswith("zone air temperature"):
            return int(i)
    for i, name in enumerate(observation_names):
        if "zone air temperature" in str(name).lower():
            return int(i)
    raise RuntimeError("Could not find a 'Zone Air Temperature' entry in observation_names")


def _find_action_index(
    action_names: list[str], component_type: str, control_type: str
) -> int | None:
    ct = component_type.strip().lower()
    ctrl = control_type.strip().lower()
    for i, name in enumerate(action_names):
        parts = str(name).split("::")
        if len(parts) < 2:
            continue
        if parts[0].strip().lower() == ct and parts[1].strip().lower() == ctrl:
            return int(i)
    return None


def _plot_timeseries(
    *,
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    out_path: Path,
    title: str,
    ylabel: str,
    hlines: list[tuple[float, str]] | None = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning(f"matplotlib not available; skipping plot {out_path.name}: {e}")
        return False

    plt.figure(figsize=(12, 5))
    for c in y_cols:
        if c in df.columns:
            plt.plot(df[x].to_numpy(), df[c].to_numpy(), label=c, linewidth=1.2)
    if hlines:
        for y, lbl in hlines:
            plt.axhline(y=y, linestyle="--", linewidth=1.0, label=lbl)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    if len(y_cols) <= 12:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


@hydra.main(version_base=None, config_path="../configs", config_name="hvac_actuator_on_off")
def main(cfg) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    run_dir = Path.cwd()
    paths = _make_rollout_paths(run_dir)

    # Save resolved config for reproducibility
    with paths.config_path.open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    eplus_output_dir = run_dir / "eplus_outputs"
    env = make_env(config=cfg, eplus_output_dir=str(eplus_output_dir))

    obs_names = _require_env_metadata_list_str(env, "observation_names")
    act_names = _require_env_metadata_list_str(env, "action_names")

    # Print action bounds early for debugging (especially inferred bounds from sizing outputs).
    try:
        if hasattr(env, "action_space") and hasattr(env.action_space, "low") and hasattr(
            env.action_space, "high"
        ):
            lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
            highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
            print("\n=== Action bounds (low/high) ===", flush=True)
            for i, name in enumerate(act_names):
                if i < len(lows) and i < len(highs):
                    print(f"{i:3d} {name}: [{lows[i]:.3f}, {highs[i]:.3f}]", flush=True)
            print("=== End action bounds ===\n", flush=True)
    except Exception as e:
        logger.warning(f"Could not print action bounds: {e}")

    control_mode = str(getattr(cfg.env, "control_mode", "")).strip()
    if control_mode == "sensible_load":
        temp_idx = _find_first_zone_air_temp_index(obs_names)
        # Select sensible-load policy implementation based on which parameters are
        # present in the loaded policy config.
        if hasattr(cfg.policy, "kp") and hasattr(cfg.policy, "ki") and hasattr(cfg.policy, "kd"):
            base_policy = PIDSensibleLoadPolicy(
                target_temp_c=float(cfg.policy.target_temp_c),
                deadband_c=float(cfg.policy.deadband_c),
                temp_obs_index=int(temp_idx),
                kp=float(cfg.policy.kp),
                ki=float(cfg.policy.ki),
                kd=float(cfg.policy.kd),
                q_heat_max_w=float(getattr(cfg.policy, "q_heat_max_w", 20000.0)),
                q_cool_max_w=float(getattr(cfg.policy, "q_cool_max_w", 20000.0)),
                dt_s=float(getattr(cfg.policy, "dt_s", 1.0)),
                integral_min=float(getattr(cfg.policy, "integral_min", -100000.0)),
                integral_max=float(getattr(cfg.policy, "integral_max", 100000.0)),
            )
        elif hasattr(cfg.policy, "respond_step_w") and hasattr(cfg.policy, "trim_step_w"):
            base_policy = TrimAndRespondSensibleLoadPolicy(
                target_temp_c=float(cfg.policy.target_temp_c),
                deadband_c=float(cfg.policy.deadband_c),
                temp_obs_index=int(temp_idx),
                respond_step_w=float(cfg.policy.respond_step_w),
                trim_step_w=float(cfg.policy.trim_step_w),
                q_heat_max_w=float(getattr(cfg.policy, "q_heat_max_w", 20000.0)),
                q_cool_max_w=float(getattr(cfg.policy, "q_cool_max_w", 20000.0)),
            )
        else:
            # Default: simple on/off controller (existing behavior)
            base_policy = OnOffSensibleLoadPolicy(
                target_temp_c=float(cfg.policy.target_temp_c),
                deadband_c=float(cfg.policy.deadband_c),
                q_heat_w=float(getattr(cfg.policy, "q_heat_w", 15000.0)),
                q_cool_w=float(getattr(cfg.policy, "q_cool_w", 15000.0)),
                temp_obs_index=int(temp_idx),
            )
        idx_load = _find_action_index(act_names, "Unitary HVAC", "Sensible Load Request")
        if idx_load is None:
            raise RuntimeError(
                "control_mode=sensible_load requires an action named "
                "'Unitary HVAC::Sensible Load Request::<component>'"
            )
        idx_avail = _find_action_index(act_names, "AirLoopHVAC", "Availability Status")
        idx_heat_sp = _find_action_index(act_names, "Zone Temperature Control", "Heating Setpoint")
        idx_cool_sp = _find_action_index(act_names, "Zone Temperature Control", "Cooling Setpoint")
    elif control_mode == "hvac_actuators":
        policy = HVACActuatorOnOffPolicy.from_env_metadata(
            env_metadata=env.metadata,  # type: ignore[arg-type]
            target_temp_c=float(cfg.policy.target_temp_c),
            deadband_c=float(cfg.policy.deadband_c),
            fan_mass_flow_kg_s=float(cfg.policy.fan_mass_flow_kg_s),
            terminal_mass_flow_kg_s=float(cfg.policy.terminal_mass_flow_kg_s),
            coil_speed_heat=float(cfg.policy.coil_speed_heat),
            coil_speed_cool=float(cfg.policy.coil_speed_cool),
            supplemental_stage_heat=float(cfg.policy.supplemental_stage_heat),
            q_heat_w=float(getattr(cfg.policy, "q_heat_w", 0.0)),
            q_cool_w=float(getattr(cfg.policy, "q_cool_w", 0.0)),
            availability_on=float(cfg.policy.availability_on),
            availability_off=float(cfg.policy.availability_off),
        )
    else:
        raise ValueError(
            "env.control_mode must be one of: 'sensible_load', 'hvac_actuators'. "
            f"Got: {control_mode!r}"
        )

    max_steps = int(getattr(cfg.env, "max_steps", 2000))
    n_episodes = int(getattr(cfg, "n_episodes", 1))
    target = float(cfg.policy.target_temp_c)
    deadband = float(cfg.policy.deadband_c)

    all_rows: list[dict[str, float]] = []

    for ep in range(n_episodes):
        obs, _info = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # Policy outputs float actuator commands (physical units).
            if control_mode == "sensible_load":
                # Build a full action vector (may include AirLoopHVAC availability).
                load_cmd, _ = base_policy.predict(obs, deterministic=True)
                tz = float(np.asarray(obs, dtype=float).reshape(-1)[temp_idx])
                action_cmd = np.zeros((len(act_names),), dtype=float)
                action_cmd[idx_load] = float(np.asarray(load_cmd, dtype=float).reshape(-1)[0])
                if idx_avail is not None:
                    # Force CycleOn to keep the HVAC system available while using load request.
                    action_cmd[idx_avail] = float(getattr(cfg.policy, "availability_on", 2.0))
                # Ensure the thermostat logic does not block the equipment manager from
                # evaluating heating/cooling operation. We set a *very small* call by
                # nudging setpoints around the current zone temperature.
                #
                # This is not used as a "temperature controller"; the magnitude is still
                # commanded via the sensible-load request. It's only a mode-enabler.
                q = float(action_cmd[idx_load])
                if idx_heat_sp is not None and idx_cool_sp is not None:
                    if q > 0.0:
                        # Heating call: ask for slightly more than current temp.
                        action_cmd[idx_heat_sp] = tz + 0.5
                        action_cmd[idx_cool_sp] = tz + 100.0
                    elif q < 0.0:
                        # Cooling call: ask for slightly less than current temp.
                        action_cmd[idx_heat_sp] = tz - 100.0
                        action_cmd[idx_cool_sp] = tz - 0.5
                    else:
                        # No load: wide deadband.
                        action_cmd[idx_heat_sp] = tz - 100.0
                        action_cmd[idx_cool_sp] = tz + 100.0
            else:
                action_cmd, _ = policy.predict(obs, deterministic=True)

            # Env may be configured with a MultiDiscrete action space; if so, convert
            # float commands -> discrete indices using the same discretization logic.
            step_action = action_cmd
            if isinstance(env.action_space, MultiDiscrete):
                hvac_actuators = env.metadata.get("hvac_actuators", [])
                if not isinstance(hvac_actuators, list) or not hvac_actuators:
                    raise RuntimeError(
                        "Expected env.metadata['hvac_actuators'] to be a non-empty list "
                        "when using MultiDiscrete HVAC actuator control."
                    )
                n_bins_continuous = int(getattr(cfg.env, "n_bins_continuous", 21))
                md = hvac_actuators_multidiscrete_transform(
                    hvac_actuators, n_bins_continuous=n_bins_continuous
                )
                step_action = md(action_cmd)

            obs2, reward, terminated, truncated, _info = env.step(step_action)

            row: dict[str, float] = {
                "episode": float(ep),
                "step": float(step),
                "reward": float(reward),
            }

            obs_arr = np.asarray(obs2, dtype=float).reshape(-1)
            for i, name in enumerate(obs_names):
                if i < len(obs_arr):
                    row[f"obs::{name}"] = float(obs_arr[i])

            # Log the *commanded actuator values* (not discrete indices) for readability.
            act_arr = np.asarray(action_cmd, dtype=float).reshape(-1)
            for i, name in enumerate(act_names):
                if i < len(act_arr):
                    row[f"act::{name}"] = float(act_arr[i])

            all_rows.append(row)

            obs = obs2
            done = bool(terminated or truncated)
            step += 1

        last_mode = (
            getattr(base_policy, "last_mode", "n/a")
            if control_mode == "sensible_load"
            else getattr(policy, "last_mode", "n/a")
        )
        logger.info(f"episode={ep} steps={step} last_mode={last_mode} saved_rows={len(all_rows)}")

    df = pd.DataFrame(all_rows)
    df.to_csv(paths.csv_path, index=False)

    # Also store dense arrays for fast loading
    obs_cols = [c for c in df.columns if c.startswith("obs::")]
    act_cols = [c for c in df.columns if c.startswith("act::")]
    np.savez_compressed(
        paths.npz_path,
        episode=df["episode"].to_numpy(dtype=np.int32),
        step=df["step"].to_numpy(dtype=np.int32),
        reward=df["reward"].to_numpy(dtype=float),
        obs_names=np.asarray(obs_cols, dtype=object),
        act_names=np.asarray(act_cols, dtype=object),
        obs=df[obs_cols].to_numpy(dtype=float) if obs_cols else np.zeros((len(df), 0)),
        act=df[act_cols].to_numpy(dtype=float) if act_cols else np.zeros((len(df), 0)),
    )

    # --- Graphs ---
    # Temperature plot: all zone temps + outdoor temp (if present)
    zone_temp_cols = [c for c in obs_cols if "zone air temperature" in c.lower()]
    outdoor_temp_cols = [c for c in obs_cols if c.lower().endswith("outdoor_temperature")]
    temp_cols = zone_temp_cols + outdoor_temp_cols
    _plot_timeseries(
        df=df,
        x="step",
        y_cols=temp_cols[:25],  # cap for readability
        out_path=paths.plot_temperature_path,
        title=f"Zone Temperatures (target={target:.1f}C deadband=±{deadband:.1f}C)",
        ylabel="Temperature [C]",
        hlines=[
            (target, "target"),
            (target - deadband, "target-deadband"),
            (target + deadband, "target+deadband"),
        ],
    )

    # Energy plot
    energy_cols = [c for c in obs_cols if c.lower().endswith("energy_electricity")] + [
        c for c in obs_cols if c.lower().endswith("energy_gas")
    ]
    _plot_timeseries(
        df=df,
        x="step",
        y_cols=energy_cols,
        out_path=paths.plot_energy_path,
        title="HVAC Energy (per-area, per-timestep)",
        ylabel="Wh / m2 / timestep",
    )

    # Actuators plot
    _plot_timeseries(
        df=df,
        x="step",
        y_cols=act_cols[:25],  # cap for readability
        out_path=paths.plot_actuators_path,
        title="HVAC Actuator Commands",
        ylabel="Actuator value (varies by actuator)",
    )

    logger.info(f"Saved raw rollout CSV: {paths.csv_path}")
    logger.info(f"Saved raw rollout NPZ: {paths.npz_path}")
    logger.info(f"Saved plots under: {paths.out_dir}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()


