"""
ASHRAE Guideline 36 (Section 5.18) controller for single-zone unitary systems.

Each conditioned zone has an independent PSZ with two actuators:
  - Fan Air Mass Flow Rate [kg/s]
  - Supply Air Temperature setpoint [°C]

Control strategy per G36 §5.18.4:
  1. Two PI controllers produce normalised heating (uHeat) and cooling (uCool)
     demand signals in [0, 1].
  2. Fan flow and SAT are mapped *directly* from those signals via
     piecewise-linear functions — no trim-and-respond delay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from b2b.baselines.common import (
    find_day_of_week_index,
    find_time_of_day_index,
    find_zone_air_temp_index_for_zone,
    require_env_metadata_list_str,
)

logger = logging.getLogger(__name__)


# ── tiny PI with back-calculation anti-windup ──────────────────────────


@dataclass
class _PIState:
    integral: float = 0.0


def _pi_signal(error: float, state: _PIState, kp: float, ki: float) -> float:
    """PI controller producing a signal clamped to [0, 1].

    *error* should be non-negative (caller selects the sign convention).
    Back-calculation anti-windup with a non-negative floor on the integral
    prevents the deep wind-down that causes output to collapse when a
    large error decreases slightly.
    """
    state.integral += error
    raw = kp * error + ki * state.integral
    output = max(0.0, min(1.0, raw))
    if ki > 0.0 and raw != output:
        state.integral = max(0.0, (output - kp * error) / ki)
    return output


# ── G36 §5.18.4 piecewise mappings ────────────────────────────────────


def _fan_fraction(u_heat: float, u_cool: float, min_f: float, med_f: float) -> float:
    """Return fan speed as a fraction of design-max [0, 1].

    Breakpoints follow G36 Table 5.18.4:
      Heating  0–50 %  → min_f
      Heating 50–100 % → min_f … 1.0
      Cooling  0–25 %  → min_f
      Cooling 25–50 %  → min_f … med_f
      Cooling 50–75 %  → med_f
      Cooling 75–100 % → med_f … 1.0
      Deadband          → min_f
    """
    if u_heat > 0.0:
        if u_heat <= 0.5:
            return min_f
        return min_f + (1.0 - min_f) * (u_heat - 0.5) / 0.5
    if u_cool > 0.0:
        if u_cool <= 0.25:
            return min_f
        if u_cool <= 0.50:
            return min_f + (med_f - min_f) * (u_cool - 0.25) / 0.25
        if u_cool <= 0.75:
            return med_f
        return med_f + (1.0 - med_f) * (u_cool - 0.75) / 0.25
    return min_f


def _sat_setpoint(
    u_heat: float,
    u_cool: float,
    sat_min_c: float,
    sat_max_c: float,
    sat_dead_c: float,
) -> float:
    """Return supply-air temperature setpoint [°C].

    Breakpoints follow G36 Table 5.18.4:
      Heating  0–50 %  → sat_dead … sat_max
      Heating 50–100 % → sat_max
      Cooling  0–25 %  → sat_dead
      Cooling 25–75 %  → sat_dead … sat_min
      Cooling 75–100 % → sat_min
      Deadband          → sat_dead
    """
    if u_heat > 0.0:
        if u_heat <= 0.5:
            return sat_dead_c + (sat_max_c - sat_dead_c) * u_heat / 0.5
        return sat_max_c
    if u_cool > 0.0:
        if u_cool <= 0.25:
            return sat_dead_c
        if u_cool <= 0.75:
            return sat_dead_c + (sat_min_c - sat_dead_c) * (u_cool - 0.25) / 0.5
        return sat_min_c
    return sat_dead_c


# ── per-zone state ─────────────────────────────────────────────────────


_WARMUP_JUMP_C = 3.0  # °C jump that indicates an EnergyPlus warmup reset


@dataclass
class _ZoneState:
    fan_idx: int
    sat_idx: int
    temp_obs_idx: int
    fan_max: float  # design-max kg/s (from action space upper bound)
    sat_high: float  # max SAT from action space [°C]
    heat_pi: _PIState = field(default_factory=_PIState)
    cool_pi: _PIState = field(default_factory=_PIState)
    prev_temp: float | None = None


# ── policy ─────────────────────────────────────────────────────────────


def _match_actuator_index(
    act_names: list[str],
    component_type: str,
    control_type: str,
    component_name: str,
) -> int:
    target = f"{component_type}::{control_type}::{component_name}"
    for i, name in enumerate(act_names):
        if name == target:
            return i
    raise RuntimeError(f"Could not find action for actuator: {target!r}")


class UnitaryG36Policy:
    """ASHRAE G36 §5.18 piecewise-linear controller for single-zone VAV."""

    def __init__(self, policy_cfg: Any) -> None:
        self.heating_sp_c: float = float(getattr(policy_cfg, "heating_setpoint_c", 21.0))
        self.cooling_sp_c: float = float(getattr(policy_cfg, "cooling_setpoint_c", 24.0))
        self.kp: float = float(getattr(policy_cfg, "kp", 1.0))
        self.ki: float = float(getattr(policy_cfg, "ki", 0.05))
        self.min_fan_frac: float = float(getattr(policy_cfg, "min_fan_fraction", 0.15))
        self.med_fan_frac: float = float(getattr(policy_cfg, "med_fan_fraction", 0.50))
        self.sat_min_c: float = float(getattr(policy_cfg, "sat_min_c", 13.0))
        self._sat_max_override: float | None = _opt_float(policy_cfg, "sat_max_c")
        self.availability_on: float = float(getattr(policy_cfg, "availability_on", 2.0))

        # schedule (reuse same structure as unitary_sat)
        sched_cfg = getattr(policy_cfg, "target_schedule", None)
        self._sched_enabled: bool = (
            bool(getattr(sched_cfg, "enabled", False))
            if sched_cfg is not None
            else False
        )
        if self._sched_enabled and sched_cfg is not None:
            wkd = getattr(sched_cfg, "weekend_days", [1, 7])
            self._weekend_days: set[int] = (
                {int(x) for x in wkd} if isinstance(wkd, (list, tuple)) else {1, 7}
            )
            self._weekend_c: float = float(getattr(sched_cfg, "weekend_target_c", self.heating_sp_c))
            self._weekday_c: float = float(getattr(sched_cfg, "weekday_target_c", self.heating_sp_c))
            self._setback_c: float = float(getattr(sched_cfg, "weekday_setback_target_c", self.heating_sp_c))
            self._setback_start: float = float(getattr(sched_cfg, "weekday_setback_start_hour", 9.0))
            self._setback_end: float = float(getattr(sched_cfg, "weekday_setback_end_hour", 16.0))
        else:
            self._weekend_days = {1, 7}
            self._weekend_c = self.heating_sp_c
            self._weekday_c = self.heating_sp_c
            self._setback_c = self.heating_sp_c
            self._setback_start = 9.0
            self._setback_end = 16.0

        self._zones: list[_ZoneState] = []
        self._avail_idxs: list[int] = []
        self._n_act: int = 0
        self._tod_idx: int | None = None
        self._dow_idx: int | None = None

    # ── env binding ────────────────────────────────────────────────────

    def bind_env(self, env: Any) -> None:
        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")

        lows = highs = None
        try:
            if hasattr(env, "action_space") and hasattr(env.action_space, "low"):
                lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
                highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
        except Exception:
            pass

        equipment = (
            env.metadata.get("hvac_equipment", [])
            if hasattr(env, "metadata") and isinstance(env.metadata, dict)
            else []
        )
        unitary_systems = [
            e for e in equipment
            if hasattr(e, "equipment_type") and e.equipment_type == "unitarysystem"
        ]

        self._avail_idxs = []
        self._zones = []
        seen_avail: set[int] = set()

        for sys in unitary_systems:
            fan_idx: int | None = None
            sat_idx: int | None = None
            for act in sys.actuator_descriptions():
                idx = _match_actuator_index(
                    act_names, act.component_type, act.control_type, act.component_name,
                )
                ct = act.control_type.lower()
                ctype = act.component_type.lower()
                if "fan air mass flow rate" in ct:
                    fan_idx = idx
                elif "availability" in ct and ctype.startswith("airloophvac"):
                    if idx not in seen_avail:
                        self._avail_idxs.append(idx)
                        seen_avail.add(idx)
                else:
                    sat_idx = idx

            if fan_idx is None or sat_idx is None:
                continue

            temp_idx = find_zone_air_temp_index_for_zone(obs_names, zone_name=sys.zone)

            fan_max = float(highs[fan_idx]) if highs is not None else 1.0
            sat_high = float(highs[sat_idx]) if highs is not None else 80.0

            self._zones.append(
                _ZoneState(
                    fan_idx=fan_idx,
                    sat_idx=sat_idx,
                    temp_obs_idx=temp_idx,
                    fan_max=fan_max,
                    sat_high=sat_high,
                )
            )

        if not self._zones:
            logger.warning("UnitaryG36Policy: no unitary systems discovered — policy is a no-op")

        if self._sched_enabled:
            try:
                self._tod_idx = find_time_of_day_index(obs_names)
                self._dow_idx = find_day_of_week_index(obs_names)
            except Exception:
                self._tod_idx = None
                self._dow_idx = None

        self._n_act = len(act_names)
        self.reset()

    # ── reset / schedule ───────────────────────────────────────────────

    def reset(self) -> None:
        for z in self._zones:
            z.heat_pi = _PIState()
            z.cool_pi = _PIState()
            z.prev_temp = None

    def _current_setpoints(self, obs_arr: np.ndarray) -> tuple[float, float]:
        """Return (heating_sp, cooling_sp) for the current timestep."""
        if not self._sched_enabled or self._tod_idx is None or self._dow_idx is None:
            return self.heating_sp_c, self.cooling_sp_c

        hour = float(obs_arr[self._tod_idx])
        day = int(obs_arr[self._dow_idx])

        if day in self._weekend_days:
            sp = self._weekend_c
        elif self._setback_start <= hour < self._setback_end:
            sp = self._setback_c
        else:
            sp = self._weekday_c

        gap = self.cooling_sp_c - self.heating_sp_c
        return sp, sp + gap

    # ── predict ────────────────────────────────────────────────────────

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        if not self._zones:
            raise RuntimeError("Policy not bound to an env; call bind_env() first.")

        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        heat_sp, cool_sp = self._current_setpoints(obs_arr)
        sat_dead = max(21.0, min(24.0, (heat_sp + cool_sp) / 2.0))

        action = np.zeros(self._n_act, dtype=float)

        for idx in self._avail_idxs:
            action[idx] = self.availability_on

        for z in self._zones:
            tz = float(obs_arr[z.temp_obs_idx])

            if z.prev_temp is not None and abs(tz - z.prev_temp) > _WARMUP_JUMP_C:
                z.heat_pi = _PIState()
                z.cool_pi = _PIState()
            z.prev_temp = tz

            heat_err = max(0.0, heat_sp - tz)
            cool_err = max(0.0, tz - cool_sp)

            u_heat = _pi_signal(heat_err, z.heat_pi, self.kp, self.ki)
            u_cool = _pi_signal(cool_err, z.cool_pi, self.kp, self.ki)

            frac = _fan_fraction(u_heat, u_cool, self.min_fan_frac, self.med_fan_frac)
            action[z.fan_idx] = frac * z.fan_max

            sat_max = self._sat_max_override if self._sat_max_override is not None else z.sat_high
            action[z.sat_idx] = _sat_setpoint(
                u_heat, u_cool, self.sat_min_c, sat_max, sat_dead,
            )

        return action, None

    # ── metrics ────────────────────────────────────────────────────────

    def step_metrics(self, obs: Any, *, action: np.ndarray) -> dict[str, float]:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        heat_sp, cool_sp = self._current_setpoints(obs_arr)
        temps: list[float] = []
        for z in self._zones:
            if z.temp_obs_idx < len(obs_arr):
                temps.append(float(obs_arr[z.temp_obs_idx]))
        metrics: dict[str, float] = {
            "heating_setpoint_c": heat_sp,
            "cooling_setpoint_c": cool_sp,
        }
        if temps:
            metrics["mean_zone_temp_c"] = float(np.mean(temps))
            metrics["min_zone_temp_c"] = float(np.min(temps))
            metrics["max_zone_temp_c"] = float(np.max(temps))
        return metrics


# ── helpers ────────────────────────────────────────────────────────────


def _opt_float(cfg: Any, key: str) -> float | None:
    v = getattr(cfg, key, None)
    if v is None:
        return None
    return float(v)
