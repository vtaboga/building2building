from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algorithms.utils import make_env, plot_timeseries

logger = logging.getLogger(__name__)


class Policy(ABC):
    @abstractmethod
    def predict(self, observation: Any, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        raise NotImplementedError


class ConstantPolicy(Policy):
    """Dummy policy that always returns the same action."""

    def __init__(self, actions: Any = None):
        self.actions: Any = actions

    def set_actions(self, actions: Any) -> None:
        print(f"setting actions: {actions}")
        self.actions = actions

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, None]:
        return self.actions, None


class ConstantSetPoints(Policy):
    def __init__(self, heating_setpoint: float, delta_setpoint: float):
        self.heating_setpoint = heating_setpoint
        self.delta_setpoint = delta_setpoint

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[list[float], None]:
        action = [self.heating_setpoint, self.delta_setpoint]
        return action, None


@dataclass(frozen=True)
class SensibleLoadBounds:
    """
    Helper to convert between absolute sensible-load request [W] and normalized percent [-1, 1].

    Convention:
    - pct in [0, 1] maps to heating in [0, high_w]
    - pct in [-1, 0) maps to cooling in [low_w, 0)
    """

    low_w: float
    high_w: float

    def __post_init__(self) -> None:
        lo = float(self.low_w)
        hi = float(self.high_w)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite load bounds: low_w={lo}, high_w={hi}")
        # Allow one-sided or degenerate bounds (e.g. unconditioned buildings),
        # but preserve sign convention when present.
        if hi < 0.0:
            raise ValueError(f"Expected non-negative heating bound (high_w), got {hi}")
        if lo > 0.0:
            raise ValueError(f"Expected non-positive cooling bound (low_w), got {lo}")

    def w_to_pct(self, q_w: float) -> float:
        q = float(q_w)
        if q >= 0.0:
            hi = float(self.high_w)
            if hi <= 0.0:
                return 0.0
            return float(np.clip(q / hi, 0.0, 1.0))
        lo = float(self.low_w)
        if lo >= 0.0:
            return 0.0
        denom = abs(lo)
        if denom <= 0.0:
            return 0.0
        return float(np.clip(q / denom, -1.0, 0.0))

    def pct_to_w(self, pct: float) -> float:
        p = float(np.clip(float(pct), -1.0, 1.0))
        if p >= 0.0:
            hi = float(self.high_w)
            if hi <= 0.0:
                return 0.0
            return float(p * hi)
        # p < 0 => negative cooling request
        lo = float(self.low_w)
        if lo >= 0.0:
            return 0.0
        return float(p * abs(lo))


@dataclass(frozen=True)
class OnOffSensibleLoadPolicy(Policy):
    """
    Simple thermostat-like on/off controller for a 1D action space:
    "Unitary HVAC :: Sensible Load Request" expressed as normalized percent [-1, 1].

    Positive = heating request, negative = cooling request, 0 = off.
    """

    target_temp_c: float
    deadband_c: float
    q_heat_pct: float
    q_cool_pct: float
    temp_obs_index: int
    last_mode: str = "off"

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        tz = float(obs[self.temp_obs_index])

        if tz < self.target_temp_c - self.deadband_c:
            object.__setattr__(self, "last_mode", "heat")
            return np.asarray([float(self.q_heat_pct)], dtype=float), None
        if tz > self.target_temp_c + self.deadband_c:
            object.__setattr__(self, "last_mode", "cool")
            return np.asarray([-abs(float(self.q_cool_pct))], dtype=float), None

        object.__setattr__(self, "last_mode", "off")
        return np.asarray([0.0], dtype=float), None


@dataclass(slots=True)
class PIDSensibleLoadPolicy(Policy):
    """
    PID controller for a 1D action space: "Unitary HVAC :: Sensible Load Request"
    expressed as normalized percent [-1, 1] (mapped to W via inferred actuator bounds).

    Sign convention:
    - Positive output requests heating
    - Negative output requests cooling

    Notes:
    - This policy assumes the observation contains a zone air temperature at `temp_obs_index`.
    - `dt_s` is the control interval in seconds. If you're stepping at fixed timestep,
      leave it constant. If unknown, leaving it at 1.0 still yields a stable controller
      after retuning gains.
    """

    target_temp_c: float
    deadband_c: float
    temp_obs_index: int

    # PID gains on error e = target - tz
    kp: float
    ki: float
    kd: float

    # Output saturation (percent in [0, 1])
    q_heat_max_pct: float
    q_cool_max_pct: float

    # Bounds used to normalize (W <-> pct)
    bounds: SensibleLoadBounds

    # Control interval
    dt_s: float = 1.0

    # Anti-windup: clamp integral of error (in C*s)
    integral_min: float = -1e6
    integral_max: float = 1e6

    # State
    integral: float = 0.0
    prev_error: float | None = None
    last_mode: str = "off"

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        tz = float(obs[self.temp_obs_index])
        error = float(self.target_temp_c - tz)

        # Deadband: explicitly off. Also bleed integral slowly to avoid latch-up.
        if abs(error) <= float(self.deadband_c):
            self.integral *= 0.9
            object.__setattr__(self, "last_mode", "off")
            object.__setattr__(self, "prev_error", error)
            return np.asarray([0.0], dtype=float), None

        dt = float(self.dt_s)
        if dt <= 0:
            raise ValueError("dt_s must be > 0")

        # Integrate error
        self.integral = float(self.integral + error * dt)
        self.integral = float(np.clip(self.integral, self.integral_min, self.integral_max))

        # Derivative
        if self.prev_error is None:
            d_error = 0.0
        else:
            d_error = float((error - self.prev_error) / dt)

        u_w = float(self.kp * error + self.ki * self.integral + self.kd * d_error)
        u_pct = float(self.bounds.w_to_pct(u_w))

        # Saturate to requested percent caps (still within [-1, 1]).
        u_pct = float(np.clip(u_pct, -abs(self.q_cool_max_pct), abs(self.q_heat_max_pct)))

        object.__setattr__(self, "prev_error", error)
        object.__setattr__(self, "last_mode", "heat" if u_pct > 0 else "cool")
        return np.asarray([u_pct], dtype=float), None


@dataclass(slots=True)
class TrimAndRespondSensibleLoadPolicy(Policy):
    """
    Trim-and-Respond controller for a sensible load request actuator.

    Control idea:
    - Maintain an internal "current load request" (W).
    - If temperature is outside deadband, "respond" by stepping load up/down.
    - If temperature is inside deadband, "trim" the load back toward zero.

    This is often easier to tune than PID for systems with long time constants.
    """

    target_temp_c: float
    deadband_c: float
    temp_obs_index: int

    # Step sizes (percent) for response and trim
    respond_step_pct: float
    trim_step_pct: float

    # Output saturation (percent)
    q_heat_max_pct: float
    q_cool_max_pct: float

    # State
    current_load_pct: float = 0.0
    last_mode: str = "off"

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        tz = float(obs[self.temp_obs_index])
        err = float(self.target_temp_c - tz)

        if abs(err) <= float(self.deadband_c):
            # Trim toward 0
            if self.current_load_pct > 0.0:
                self.current_load_pct = max(
                    0.0,
                    float(self.current_load_pct) - abs(float(self.trim_step_pct)),
                )
            elif self.current_load_pct < 0.0:
                self.current_load_pct = min(
                    0.0,
                    float(self.current_load_pct) + abs(float(self.trim_step_pct)),
                )
            object.__setattr__(
                self,
                "last_mode",
                "off" if float(self.current_load_pct) == 0.0 else self.last_mode,
            )
            return np.asarray([float(self.current_load_pct)], dtype=float), None

        # Respond: step in the direction of the error
        if err > 0.0:
            step_pct = abs(float(self.respond_step_pct))
            self.current_load_pct = float(self.current_load_pct + step_pct)
            self.current_load_pct = float(
                min(self.current_load_pct, abs(float(self.q_heat_max_pct)))
            )
            object.__setattr__(self, "last_mode", "heat")
        else:
            step_pct = abs(float(self.respond_step_pct))
            self.current_load_pct = float(self.current_load_pct - step_pct)
            self.current_load_pct = float(
                max(self.current_load_pct, -abs(float(self.q_cool_max_pct)))
            )
            object.__setattr__(self, "last_mode", "cool")

        return np.asarray([float(self.current_load_pct)], dtype=float), None


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


def _make_policy(cfg: DictConfig, *, temp_obs_index: int, bounds: SensibleLoadBounds) -> Policy:
    policy_type = str(getattr(cfg.policy, "type"))
    if policy_type == "pid":
        # Back-compat: config gains are in W; this policy outputs pct by normalizing via bounds.
        if hasattr(cfg.policy, "q_heat_max_pct") and hasattr(cfg.policy, "q_cool_max_pct"):
            q_heat_max_pct = float(getattr(cfg.policy, "q_heat_max_pct"))
            q_cool_max_pct = float(getattr(cfg.policy, "q_cool_max_pct"))
        else:
            q_heat_max_w = float(getattr(cfg.policy, "q_heat_max_w", 20000.0))
            q_cool_max_w = float(getattr(cfg.policy, "q_cool_max_w", 20000.0))
            q_heat_max_pct = float(bounds.w_to_pct(abs(q_heat_max_w)))
            q_cool_max_pct = float(abs(bounds.w_to_pct(-abs(q_cool_max_w))))
        return PIDSensibleLoadPolicy(
            target_temp_c=float(cfg.policy.target_temp_c),
            deadband_c=float(cfg.policy.deadband_c),
            temp_obs_index=int(temp_obs_index),
            kp=float(cfg.policy.kp),
            ki=float(cfg.policy.ki),
            kd=float(cfg.policy.kd),
            q_heat_max_pct=float(q_heat_max_pct),
            q_cool_max_pct=float(q_cool_max_pct),
            bounds=bounds,
            dt_s=float(getattr(cfg.policy, "dt_s", 1.0)),
            integral_min=float(getattr(cfg.policy, "integral_min", -100000.0)),
            integral_max=float(getattr(cfg.policy, "integral_max", 100000.0)),
        )
    if policy_type == "trim_and_respond":
        # Back-compat: accept either pct (preferred) or W (converted via inferred bounds).
        if hasattr(cfg.policy, "respond_step_pct") and hasattr(cfg.policy, "trim_step_pct"):
            respond_step_pct = float(getattr(cfg.policy, "respond_step_pct"))
            trim_step_pct = float(getattr(cfg.policy, "trim_step_pct"))
        else:
            respond_step_w = float(getattr(cfg.policy, "respond_step_w"))
            trim_step_w = float(getattr(cfg.policy, "trim_step_w"))
            respond_step_pct = float(abs(bounds.w_to_pct(abs(respond_step_w))))
            trim_step_pct = float(abs(bounds.w_to_pct(abs(trim_step_w))))

        if hasattr(cfg.policy, "q_heat_max_pct") and hasattr(cfg.policy, "q_cool_max_pct"):
            q_heat_max_pct = float(getattr(cfg.policy, "q_heat_max_pct"))
            q_cool_max_pct = float(getattr(cfg.policy, "q_cool_max_pct"))
        else:
            q_heat_max_w = float(getattr(cfg.policy, "q_heat_max_w", 20000.0))
            q_cool_max_w = float(getattr(cfg.policy, "q_cool_max_w", 20000.0))
            q_heat_max_pct = float(bounds.w_to_pct(abs(q_heat_max_w)))
            q_cool_max_pct = float(abs(bounds.w_to_pct(-abs(q_cool_max_w))))
        return TrimAndRespondSensibleLoadPolicy(
            target_temp_c=float(cfg.policy.target_temp_c),
            deadband_c=float(cfg.policy.deadband_c),
            temp_obs_index=int(temp_obs_index),
            respond_step_pct=float(respond_step_pct),
            trim_step_pct=float(trim_step_pct),
            q_heat_max_pct=float(q_heat_max_pct),
            q_cool_max_pct=float(q_cool_max_pct),
        )
    if policy_type == "on_off":
        # Back-compat: accept either pct (preferred) or W (converted via inferred bounds).
        if hasattr(cfg.policy, "q_heat_pct") and hasattr(cfg.policy, "q_cool_pct"):
            q_heat_pct = float(getattr(cfg.policy, "q_heat_pct"))
            q_cool_pct = float(getattr(cfg.policy, "q_cool_pct"))
        else:
            q_heat_w = float(getattr(cfg.policy, "q_heat_w"))
            q_cool_w = float(getattr(cfg.policy, "q_cool_w"))
            q_heat_pct = float(bounds.w_to_pct(abs(q_heat_w)))
            q_cool_pct = float(abs(bounds.w_to_pct(-abs(q_cool_w))))
        return OnOffSensibleLoadPolicy(
            target_temp_c=float(cfg.policy.target_temp_c),
            deadband_c=float(cfg.policy.deadband_c),
            q_heat_pct=float(q_heat_pct),
            q_cool_pct=float(q_cool_pct),
            temp_obs_index=int(temp_obs_index),
        )
    raise ValueError(f"Unknown policy.type: {policy_type}")


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

        idx_load = find_action_index(act_names, "Unitary HVAC", "Sensible Load Request")
        print(f"idx_load: {idx_load}")

        if idx_load is None and find_action_index(act_names, "Zone Temperature Control", "Heating Setpoint") is None:
            raise RuntimeError(
                "Baseline controller requires either a 'Unitary HVAC::Sensible Load Request::<component>' "
                "actuator or at least one Zone Temperature Control setpoint actuator."
            )

        load_bounds: SensibleLoadBounds | None = None
        policy: Policy | None = None
        if idx_load is not None:
            if not hasattr(env, "action_space") or not hasattr(env.action_space, "low") or not hasattr(
                env.action_space, "high"
            ):
                raise RuntimeError("env.action_space.low/high is required for action normalization")
            lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
            highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
            if idx_load >= len(lows) or idx_load >= len(highs):
                raise RuntimeError("idx_load out of bounds for env.action_space.low/high")
            load_bounds = SensibleLoadBounds(
                low_w=float(lows[idx_load]),
                high_w=float(highs[idx_load]),
            )

            controlled_zones = None
            try:
                controlled_zones = require_env_metadata_list_str(env, "controlled_zones")
            except Exception:
                controlled_zones = None
            temp_idx = find_controlled_zone_air_temp_index(
                obs_names, controlled_zones=controlled_zones
            )
            policy = _make_policy(cfg, temp_obs_index=temp_idx, bounds=load_bounds)
        else:
            # Baseboard-only mode: select a controlled-zone temperature for thermostat logic.
            controlled_zones = None
            try:
                controlled_zones = require_env_metadata_list_str(env, "controlled_zones")
            except Exception:
                controlled_zones = None
            temp_idx = find_controlled_zone_air_temp_index(
                obs_names, controlled_zones=controlled_zones
            )

        idx_avail = find_action_index(act_names, "AirLoopHVAC", "Availability Status")

        idx_heat_sp = find_action_index(act_names, "Zone Temperature Control", "Heating Setpoint")
        idx_cool_sp = find_action_index(act_names, "Zone Temperature Control", "Cooling Setpoint")

        max_steps = int(getattr(cfg.env, "max_steps"))
        n_episodes = int(getattr(cfg, "n_episodes"))
        target = float(cfg.policy.target_temp_c)
        deadband = float(cfg.policy.deadband_c)

        all_rows: list[dict[str, float]] = []

        for ep in range(n_episodes):
            obs, _info = env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                tz = float(np.asarray(obs, dtype=float).reshape(-1)[temp_idx])
                target = float(cfg.policy.target_temp_c)
                deadband = float(cfg.policy.deadband_c)

                forced_mode = getattr(cfg.policy, "forced_mode", None)
                if forced_mode is not None:
                    forced_mode = str(forced_mode).strip().lower()

                # Determine requested mode:
                # - if forced, use that
                # - else if we have a sensible-load actuator, use the policy output (pct sign)
                # - else use a simple heat/cool/off thermostat based on target+deadband
                load_pct = 0.0
                if forced_mode in ("heat", "cool", "off"):
                    mode = forced_mode
                elif policy is not None:
                    load_pct_cmd, _ = policy.predict(obs, deterministic=True)
                    load_pct = float(np.asarray(load_pct_cmd, dtype=float).reshape(-1)[0])
                    mode = "heat" if load_pct > 0.0 else ("cool" if load_pct < 0.0 else "off")
                else:
                    if tz < target - deadband:
                        mode = "heat"
                    elif tz > target + deadband:
                        mode = "cool"
                    else:
                        mode = "off"

                action_cmd = np.zeros((len(act_names),), dtype=float)
                if idx_load is not None and load_bounds is not None:
                    action_cmd[idx_load] = float(load_bounds.pct_to_w(load_pct))

                if idx_avail is not None:
                    # Force CycleOn to keep the HVAC system available while using load request.
                    action_cmd[idx_avail] = float(getattr(cfg.policy, "availability_on", 2.0))

                # Zone setpoint actuators: actions are interpreted as ΔT from current zone air temp
                # (via SetpointDeltaActionWrapper in create_simulator).
                #
                # Therefore here we emit *deltas*, not absolute temperatures.
                if idx_heat_sp is not None and idx_cool_sp is not None:
                    if mode == "heat":
                        action_cmd[idx_heat_sp] = 0.5
                        action_cmd[idx_cool_sp] = 30.0
                    elif mode == "cool":
                        action_cmd[idx_heat_sp] = -30.0
                        action_cmd[idx_cool_sp] = -0.5
                    else:
                        action_cmd[idx_heat_sp] = -30.0
                        action_cmd[idx_cool_sp] = 30.0

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
                # Also log normalized sensible load request (pct) for easier interpretation.
                if idx_load is not None:
                    row[f"act_pct::{act_names[idx_load]}"] = float(load_pct)

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