from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algorithms.utils import make_env, plot_timeseries

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


def make_rollout_paths(run_dir: Path) -> RolloutPaths:
    run_dir.mkdir(parents=True, exist_ok=True)
    return RolloutPaths(
        out_dir=run_dir,
        csv_path=run_dir / "rollout.csv",
        npz_path=run_dir / "rollout.npz",
        config_path=run_dir / "config_resolved.json",
        plot_temperature_path=run_dir / "temperature.png",
        plot_energy_path=run_dir / "energy.png",
        plot_actuators_path=run_dir / "actuators.png",
    )


def require_env_metadata_list_str(env: Any, key: str) -> list[str]:
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("env.metadata missing")
    raw = env.metadata.get(key)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"env.metadata['{key}'] must be a list[str]")
    return raw


def find_first_zone_air_temp_index(observation_names: list[str]) -> int:
    for i, name in enumerate(observation_names):
        if str(name).lower().startswith("zone air temperature"):
            return int(i)
    for i, name in enumerate(observation_names):
        if "zone air temperature" in str(name).lower():
            return int(i)
    raise RuntimeError("Could not find a 'Zone Air Temperature' entry in observation_names")


def find_controlled_zone_air_temp_index(
    observation_names: list[str],
    *,
    controlled_zones: list[str] | None,
) -> int:
    """
    Prefer a Zone Air Temperature observation that corresponds to a controlled zone.

    This avoids accidentally controlling a temperature from an uncontrolled zone
    (e.g., garage) when the HVAC does not serve that zone.
    """
    if not controlled_zones:
        return find_first_zone_air_temp_index(observation_names)

    controlled = {str(z).strip().lower() for z in controlled_zones if str(z).strip()}
    if not controlled:
        return find_first_zone_air_temp_index(observation_names)

    prefix = "zone air temperature"
    for i, name in enumerate(observation_names):
        s = str(name).strip()
        sl = s.lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if zone_part in controlled:
            return int(i)

    # Fallback: best-effort substring match (for ontologies that decorate zone names).
    for i, name in enumerate(observation_names):
        sl = str(name).strip().lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if any(z in zone_part or zone_part in z for z in controlled):
            return int(i)

    return find_first_zone_air_temp_index(observation_names)


def find_action_index(action_names: list[str], component_type: str, control_type: str) -> int | None:
    ct = component_type.strip().lower()
    ctrl = control_type.strip().lower()
    for i, name in enumerate(action_names):
        parts = str(name).split("::")
        if len(parts) < 2:
            continue
        if parts[0].strip().lower() == ct and parts[1].strip().lower() == ctrl:
            return int(i)
    return None


def find_action_indices(
    action_names: list[str],
    *,
    component_type_prefix: str | None = None,
    control_type: str | None = None,
    component_name_contains: str | None = None,
) -> list[int]:
    """
    Find all indices in `action_names` that match simple filters on the triplet
    "{component_type}::{control_type}::{component_name}".
    """
    out: list[int] = []
    ct_prefix = component_type_prefix.strip().lower() if component_type_prefix else None
    ctrl = control_type.strip().lower() if control_type else None
    name_sub = component_name_contains.strip().lower() if component_name_contains else None

    for i, name in enumerate(action_names):
        parts = str(name).split("::")
        if len(parts) < 3:
            continue
        ct = parts[0].strip().lower()
        c = parts[1].strip().lower()
        cn = parts[2].strip().lower()
        if ct_prefix is not None and not ct.startswith(ct_prefix):
            continue
        if ctrl is not None and c != ctrl:
            continue
        if name_sub is not None and name_sub not in cn:
            continue
        out.append(i)
    return out


def run_baseline_rollout(cfg: DictConfig, *, run_dir: Path | None = None) -> RolloutPaths:
    """
    rollout a simulation with a baseline policy
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
    env = make_env(config=cfg, eplus_output_dir=str(eplus_output_dir))
    try:
        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")

        # Print action bounds for debugging.
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

        # We still pick a representative controlled-zone temperature for plotting/debugging.
        controlled_zones = None
        try:
            controlled_zones = require_env_metadata_list_str(env, "controlled_zones")
        except Exception:
            controlled_zones = None
        temp_idx = find_controlled_zone_air_temp_index(obs_names, controlled_zones=controlled_zones)

        # Control mode: fan airflow rate + node temperature setpoints + availability.
        #
        # We support two equivalent control encodings:
        # - Preferred (robust): Schedule:* / Schedule Value actuators for `B2B Node Temp SP ...`
        #   schedules used by SetpointManager:Scheduled objects.
        # - Fallback (legacy / fixtures): System Node Setpoint / Temperature Setpoint actuators.
        idx_avail = find_action_indices(
            act_names,
            component_type_prefix="airloophvac",
            control_type="availability status",
        )
        idx_fans = find_action_indices(
            act_names,
            component_type_prefix="fan",
            control_type="fan air mass flow rate",
        )
        # 1) Preferred: scheduled setpoints (Schedule Value actuators)
        idx_heat_nodes_sched = find_action_indices(
            act_names,
            component_type_prefix="schedule:",
            control_type="schedule value",
            component_name_contains="b2b node temp sp heating_coil",
        )
        idx_supp_nodes_sched = find_action_indices(
            act_names,
            component_type_prefix="schedule:",
            control_type="schedule value",
            component_name_contains="b2b node temp sp supplemental_coil",
        )
        idx_cool_nodes_sched = find_action_indices(
            act_names,
            component_type_prefix="schedule:",
            control_type="schedule value",
            component_name_contains="b2b node temp sp cooling_coil",
        )
        idx_outlet_nodes_sched = find_action_indices(
            act_names,
            component_type_prefix="schedule:",
            control_type="schedule value",
            component_name_contains="b2b node temp sp unitary_outlet",
        )

        # 2) Fallback: direct system node setpoint actuators (fixtures / legacy)
        idx_heat_nodes_node = find_action_indices(
            act_names,
            component_type_prefix="system node setpoint",
            control_type="temperature setpoint",
            component_name_contains="heating coil node",
        )
        idx_supp_nodes_node = find_action_indices(
            act_names,
            component_type_prefix="system node setpoint",
            control_type="temperature setpoint",
            component_name_contains="supplemental coil node",
        )
        idx_cool_nodes_node = find_action_indices(
            act_names,
            component_type_prefix="system node setpoint",
            control_type="temperature setpoint",
            component_name_contains="cooling coil node",
        )
        idx_outlet_nodes_node: list[int] = []
        idx_all_node_setpoints = find_action_indices(
            act_names,
            component_type_prefix="system node setpoint",
            control_type="temperature setpoint",
        )
        for i in idx_all_node_setpoints:
            name = str(act_names[int(i)]).lower()
            if "node" in name and "coil node" not in name:
                idx_outlet_nodes_node.append(int(i))

        # Choose scheduled if present, otherwise fallback to node actuators.
        use_scheduled = bool(
            idx_heat_nodes_sched or idx_supp_nodes_sched or idx_cool_nodes_sched or idx_outlet_nodes_sched
        )
        if use_scheduled:
            idx_heat_nodes = idx_heat_nodes_sched
            idx_supp_nodes = idx_supp_nodes_sched
            idx_cool_nodes = idx_cool_nodes_sched
            idx_outlet_nodes = idx_outlet_nodes_sched
        else:
            idx_heat_nodes = idx_heat_nodes_node
            idx_supp_nodes = idx_supp_nodes_node
            idx_cool_nodes = idx_cool_nodes_node
            idx_outlet_nodes = idx_outlet_nodes_node

        if not idx_fans or (
            not idx_heat_nodes and not idx_supp_nodes and not idx_cool_nodes and not idx_outlet_nodes
        ):
            raise RuntimeError(
                "Baseline fan/node-setpoint controller could not find required actuators in action_names. "
                "Expected at least one Fan::Fan Air Mass Flow Rate and at least one "
                "node temperature setpoint actuator (scheduled or direct) for a coil/outlet node.\n"
                f"action_names={act_names}"
            )

        max_steps = int(getattr(cfg.env, "max_steps"))
        n_episodes = int(getattr(cfg, "n_episodes"))
        # Optional (used only for plot annotation).
        target = float(getattr(cfg.policy, "target_temp_c", 21.0))
        deadband = float(getattr(cfg.policy, "deadband_c", 1.0))

        fan_mdot = float(getattr(cfg.policy, "fan_mass_flow_kg_s", 1.0))
        t_heat = float(getattr(cfg.policy, "heating_coil_setpoint_c", 35.0))
        t_supp = float(getattr(cfg.policy, "supplemental_coil_setpoint_c", 45.0))
        t_cool = float(getattr(cfg.policy, "cooling_coil_setpoint_c", 12.0))
        # Supply/outlet node setpoint (unitary system air outlet node).
        t_outlet = float(getattr(cfg.policy, "outlet_node_setpoint_c", 22.0))
        availability_on = float(getattr(cfg.policy, "availability_on", 2.0))

        all_rows: list[dict[str, float]] = []

        for ep in range(n_episodes):
            obs, _info = env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                tz = float(np.asarray(obs, dtype=float).reshape(-1)[temp_idx])
                # Optional (used only for plot annotation).
                target = float(getattr(cfg.policy, "target_temp_c", target))
                deadband = float(getattr(cfg.policy, "deadband_c", deadband))

                action_cmd = np.zeros((len(act_names),), dtype=float)

                for idx in idx_avail:
                    action_cmd[int(idx)] = availability_on
                for idx in idx_fans:
                    action_cmd[int(idx)] = fan_mdot
                for idx in idx_heat_nodes:
                    action_cmd[int(idx)] = t_heat
                for idx in idx_supp_nodes:
                    action_cmd[int(idx)] = t_supp
                for idx in idx_cool_nodes:
                    action_cmd[int(idx)] = t_cool
                for idx in idx_outlet_nodes:
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
            obs=df[obs_cols].to_numpy(dtype=float) if obs_cols else np.zeros((len(df), 0)),
            act=df[act_cols].to_numpy(dtype=float) if act_cols else np.zeros((len(df), 0)),
        )

        zone_temp_cols = [c for c in obs_cols if "zone air temperature" in c.lower()]
        outdoor_temp_cols = [c for c in obs_cols if c.lower().endswith("outdoor_temperature")]
        temp_cols = zone_temp_cols + outdoor_temp_cols
        plot_timeseries(
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
        plot_timeseries(
            df=df,
            x="step",
            y_cols=energy_cols,
            out_path=paths.plot_energy_path,
            title="HVAC Energy (per-area, per-timestep)",
            ylabel="Wh / m2 / timestep",
        )

        plot_timeseries(
            df=df,
            x="step",
            y_cols=act_cols[:25],
            out_path=paths.plot_actuators_path,
            title="HVAC Actuator Commands",
            ylabel="Actuator value (varies by actuator)",
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