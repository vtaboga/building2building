"""
Baseline controller for VAV air-loop HVAC systems.

Auto-discovers the number of air loops and zones per loop from the
environment metadata (``hvac_equipment``).  Works with any building that
uses ``VAVSystem`` equipment (e.g. OfficeMedium with 3 loops × 5 zones,
or any other VAV configuration).

Action layout per loop (discovered dynamically):
  - 1  supply-air temperature setpoint  (SAT)
  - N× zone flow fraction              (damper position)
  - N× zone heating setpoint
  - N× zone cooling setpoint

Control strategy:
  SAT  — continuous function of loop zone errors, biased toward coldest zone.
  Flow — directional PI with rate limiting per zone.
  Thermostat setpoints — per-zone heating/cooling demand signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from b2b.baselines.common import (
    find_obs_index_by_exact_name,
    find_zone_air_temp_index_for_zone,
    require_env_metadata_list_str,
)


@dataclass
class AirLoopBaselineConfig:
    """Tunable parameters for the VAV air-loop baseline controller."""

    target_temp: float = 21.0
    deadband: float = 0.5

    # --- SAT control (continuous, per-loop) ---
    sat_neutral: float = 21.0
    sat_kp: float = 4.0
    sat_min: float = 10.0
    sat_max: float = 60.0
    sat_rate_limit: float = 2.0
    outdoor_sat_gain: float = 0.2
    sat_cold_bias: float = 0.6

    # --- Flow control (directional PI, per-zone) ---
    flow_base: float = 0.35
    flow_kp: float = 0.15
    flow_ki: float = 0.025
    flow_min: float = 0.0
    flow_max: float = 1.0
    flow_rate_limit: float = 0.08
    integral_max: float = 30.0
    integral_decay: float = 0.997

    # --- Thermostat setpoint control ---
    htg_sp_base: float = 21.0
    clg_sp_base: float = 24.0
    sp_kp: float = 2.5
    sp_rate_limit: float = 0.5
    htg_sp_min: float = 10.0
    htg_sp_max: float = 30.0
    clg_sp_min: float = 18.0
    clg_sp_max: float = 40.0

    # --- Smoothing ---
    error_ema_alpha: float = 0.5


@dataclass
class _ZoneState:
    """Mutable per-zone control state."""

    zone_name: str
    temp_obs_idx: int
    flow_act_idx: int
    htg_act_idx: int
    clg_act_idx: int | None

    integral: float = 0.0
    smooth_error: float = 0.0
    prev_flow: float = 0.35
    prev_htg_sp: float = 21.0
    prev_clg_sp: float = 24.0


@dataclass
class _LoopState:
    """Mutable per-loop control state."""

    sat_act_idx: int
    zones: list[_ZoneState] = field(default_factory=list)
    prev_sat: float = 21.0


def _match_actuator_to_action_index(
    act_names: list[str],
    component_type: str,
    control_type: str,
    component_name: str,
    *,
    required: bool = True,
) -> int | None:
    target = f"{component_type}::{control_type}::{component_name}"
    for i, name in enumerate(act_names):
        if name == target:
            return i
    if required:
        raise RuntimeError(
            f"Could not find action index for actuator: {target!r}\n"
            f"Available: {act_names[:20]}..."
        )
    return None


class AirLoopSatPolicy:
    """VAV air-loop baseline with SB3-like predict() API.

    Discovers loops and zones from ``env.metadata["hvac_equipment"]``
    (``VAVSystem`` objects).
    """

    def __init__(self, policy_cfg: Any = None):
        if policy_cfg is not None:
            self.config = AirLoopBaselineConfig(
                target_temp=float(getattr(policy_cfg, "target_temp", 21.0)),
                deadband=float(getattr(policy_cfg, "deadband", 0.5)),
                sat_neutral=float(getattr(policy_cfg, "sat_neutral", 21.0)),
                sat_kp=float(getattr(policy_cfg, "sat_kp", 4.0)),
                sat_min=float(getattr(policy_cfg, "sat_min", 10.0)),
                sat_max=float(getattr(policy_cfg, "sat_max", 60.0)),
                sat_rate_limit=float(getattr(policy_cfg, "sat_rate_limit", 2.0)),
                outdoor_sat_gain=float(getattr(policy_cfg, "outdoor_sat_gain", 0.2)),
                sat_cold_bias=float(getattr(policy_cfg, "sat_cold_bias", 0.6)),
                flow_base=float(getattr(policy_cfg, "flow_base", 0.35)),
                flow_kp=float(getattr(policy_cfg, "flow_kp", 0.15)),
                flow_ki=float(getattr(policy_cfg, "flow_ki", 0.025)),
                flow_min=float(getattr(policy_cfg, "flow_min", 0.0)),
                flow_max=float(getattr(policy_cfg, "flow_max", 1.0)),
                flow_rate_limit=float(getattr(policy_cfg, "flow_rate_limit", 0.08)),
                integral_max=float(getattr(policy_cfg, "integral_max", 30.0)),
                integral_decay=float(getattr(policy_cfg, "integral_decay", 0.997)),
                htg_sp_base=float(getattr(policy_cfg, "htg_sp_base", 21.0)),
                clg_sp_base=float(getattr(policy_cfg, "clg_sp_base", 24.0)),
                sp_kp=float(getattr(policy_cfg, "sp_kp", 2.5)),
                sp_rate_limit=float(getattr(policy_cfg, "sp_rate_limit", 0.5)),
                htg_sp_min=float(getattr(policy_cfg, "htg_sp_min", 10.0)),
                htg_sp_max=float(getattr(policy_cfg, "htg_sp_max", 30.0)),
                clg_sp_min=float(getattr(policy_cfg, "clg_sp_min", 18.0)),
                clg_sp_max=float(getattr(policy_cfg, "clg_sp_max", 40.0)),
                error_ema_alpha=float(getattr(policy_cfg, "error_ema_alpha", 0.5)),
            )
        else:
            self.config = AirLoopBaselineConfig()

        self._loops: list[_LoopState] = []
        self._n_act: int = 0
        self._outdoor_temp_idx: int | None = None
        self._initialized = False

    def bind_env(self, env: Any) -> None:
        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")

        equipment = env.metadata.get("hvac_equipment", [])
        vav_systems = [
            e for e in equipment
            if hasattr(e, "equipment_type") and e.equipment_type == "vavsystem"
        ]
        if not vav_systems:
            raise RuntimeError(
                "AirLoopSatPolicy requires VAVSystem equipment in env metadata. "
                f"Found equipment types: {[type(e).__name__ for e in equipment]}"
            )

        self._outdoor_temp_idx = find_obs_index_by_exact_name(
            obs_names, name="outdoor_temperature"
        )

        self._loops = []
        for vav in vav_systems:
            sat_idx = _match_actuator_to_action_index(
                act_names,
                vav.supply_temp_setpoint.component_type,
                vav.supply_temp_setpoint.control_type,
                vav.supply_temp_setpoint.component_name,
            )
            zones: list[_ZoneState] = []
            for term in vav.terminals:
                flow_idx = _match_actuator_to_action_index(
                    act_names,
                    term.flow_fraction.component_type,
                    term.flow_fraction.control_type,
                    term.flow_fraction.component_name,
                )
                htg_idx = _match_actuator_to_action_index(
                    act_names,
                    term.heating_setpoint.component_type,
                    term.heating_setpoint.control_type,
                    term.heating_setpoint.component_name,
                )
                clg_idx = _match_actuator_to_action_index(
                    act_names,
                    term.cooling_setpoint.component_type,
                    term.cooling_setpoint.control_type,
                    term.cooling_setpoint.component_name,
                    required=False,
                )
                temp_idx = find_zone_air_temp_index_for_zone(
                    obs_names, zone_name=term.zone
                )
                zones.append(_ZoneState(
                    zone_name=term.zone,
                    temp_obs_idx=temp_idx,
                    flow_act_idx=flow_idx,
                    htg_act_idx=htg_idx,
                    clg_act_idx=clg_idx,
                    prev_flow=self.config.flow_base,
                    prev_htg_sp=self.config.htg_sp_base,
                    prev_clg_sp=self.config.clg_sp_base,
                ))
            self._loops.append(_LoopState(
                sat_act_idx=sat_idx,
                zones=zones,
                prev_sat=self.config.sat_neutral,
            ))

        self._n_act = len(act_names)
        self._initialized = False
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        for loop in self._loops:
            loop.prev_sat = cfg.sat_neutral
            for z in loop.zones:
                z.integral = 0.0
                z.smooth_error = 0.0
                z.prev_flow = cfg.flow_base
                z.prev_htg_sp = cfg.htg_sp_base
                z.prev_clg_sp = cfg.clg_sp_base
        self._initialized = False

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        if not self._loops:
            raise RuntimeError(
                "Policy is not bound to an env; call bind_env(env) first."
            )

        cfg = self.config
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        action = np.zeros(self._n_act, dtype=np.float64)

        t_outdoor = float("nan")
        if self._outdoor_temp_idx is not None and self._outdoor_temp_idx < len(obs_arr):
            t_outdoor = float(obs_arr[self._outdoor_temp_idx])

        outdoor_offset = 0.0
        if not np.isnan(t_outdoor):
            outdoor_offset = cfg.outdoor_sat_gain * (cfg.target_temp - t_outdoor)

        for loop in self._loops:
            # --- Read zone temperatures & compute smoothed errors ---
            raw_errors = np.array([
                float(obs_arr[z.temp_obs_idx]) - cfg.target_temp
                if z.temp_obs_idx < len(obs_arr) else 0.0
                for z in loop.zones
            ], dtype=np.float64)

            alpha = cfg.error_ema_alpha
            for i, z in enumerate(loop.zones):
                if not self._initialized:
                    z.smooth_error = raw_errors[i]
                else:
                    z.smooth_error = alpha * raw_errors[i] + (1 - alpha) * z.smooth_error

            errors = np.array([z.smooth_error for z in loop.zones], dtype=np.float64)

            # --- SAT: continuous function biased toward coldest zone ---
            w = cfg.sat_cold_bias
            mean_err = float(np.mean(errors))
            min_err = float(np.min(errors))
            weighted_err = w * min_err + (1 - w) * mean_err

            sat_target = cfg.sat_neutral - cfg.sat_kp * weighted_err + outdoor_offset
            sat_target = np.clip(sat_target, cfg.sat_min, cfg.sat_max)

            delta = np.clip(
                sat_target - loop.prev_sat, -cfg.sat_rate_limit, cfg.sat_rate_limit
            )
            sat = float(np.clip(loop.prev_sat + delta, cfg.sat_min, cfg.sat_max))
            action[loop.sat_act_idx] = sat
            loop.prev_sat = sat

            # --- Flow: directional PI + rate limit per zone ---
            for i, z in enumerate(loop.zones):
                z.integral *= cfg.integral_decay
                at_max = z.prev_flow >= cfg.flow_max - 0.01 and errors[i] < 0
                at_min = z.prev_flow <= cfg.flow_min + 0.01 and errors[i] > 0
                if not at_max and not at_min:
                    z.integral += errors[i]
                z.integral = float(np.clip(z.integral, -cfg.integral_max, cfg.integral_max))

                flow_target = cfg.flow_base - cfg.flow_kp * errors[i] - cfg.flow_ki * z.integral
                flow_target = float(np.clip(flow_target, cfg.flow_min, cfg.flow_max))

                delta = float(np.clip(
                    flow_target - z.prev_flow, -cfg.flow_rate_limit, cfg.flow_rate_limit
                ))
                flow = float(np.clip(z.prev_flow + delta, cfg.flow_min, cfg.flow_max))
                action[z.flow_act_idx] = flow
                z.prev_flow = flow

            # --- Thermostat setpoints per zone ---
            for i, z in enumerate(loop.zones):
                err = errors[i]

                if err < -cfg.deadband:
                    htg_target = cfg.htg_sp_base + cfg.sp_kp * (-err - cfg.deadband)
                    clg_target = cfg.clg_sp_base
                elif err > cfg.deadband:
                    htg_target = cfg.htg_sp_base
                    clg_target = cfg.clg_sp_base - cfg.sp_kp * (err - cfg.deadband)
                else:
                    htg_target = cfg.htg_sp_base
                    clg_target = cfg.clg_sp_base

                htg_target = float(np.clip(htg_target, cfg.htg_sp_min, cfg.htg_sp_max))
                clg_target = float(np.clip(clg_target, cfg.clg_sp_min, cfg.clg_sp_max))

                d_htg = float(np.clip(
                    htg_target - z.prev_htg_sp, -cfg.sp_rate_limit, cfg.sp_rate_limit
                ))
                d_clg = float(np.clip(
                    clg_target - z.prev_clg_sp, -cfg.sp_rate_limit, cfg.sp_rate_limit
                ))
                htg_sp = float(np.clip(z.prev_htg_sp + d_htg, cfg.htg_sp_min, cfg.htg_sp_max))
                clg_sp = float(np.clip(z.prev_clg_sp + d_clg, cfg.clg_sp_min, cfg.clg_sp_max))

                action[z.htg_act_idx] = htg_sp
                if z.clg_act_idx is not None:
                    action[z.clg_act_idx] = clg_sp
                z.prev_htg_sp = htg_sp
                z.prev_clg_sp = clg_sp

        self._initialized = True
        return action, None

    def step_metrics(self, obs: Any, *, action: np.ndarray) -> dict[str, float]:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        metrics: dict[str, float] = {
            "target_temp_c": float(self.config.target_temp),
        }
        all_temps: list[float] = []
        for loop in self._loops:
            for z in loop.zones:
                if z.temp_obs_idx < len(obs_arr):
                    all_temps.append(float(obs_arr[z.temp_obs_idx]))
        if all_temps:
            metrics["mean_zone_temp_c"] = float(np.mean(all_temps))
            metrics["min_zone_temp_c"] = float(np.min(all_temps))
            metrics["max_zone_temp_c"] = float(np.max(all_temps))
        return metrics
