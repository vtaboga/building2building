"""
ASHRAE Guideline 36 inspired controller for single-zone unitary systems.

Each conditioned zone has an independent Packaged Single Zone (PSZ) with two
actuators:
  - Fan Air Mass Flow Rate [kg/s]
  - Supply Air Temperature setpoint [C]

Control strategy (G36-like supervisory approximation):
  1. Zone temperature PI loop modulates fan airflow as a capacity proxy.
  2. Trim-and-Respond SAT reset adjusts supply temperature based on zone
     demand: zone too warm -> respond down (lower SAT), zone too cold ->
     respond up (raise SAT), zone satisfied -> trim toward neutral.
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


# -- PI controller ---------------------------------------------------------


@dataclass
class _PIState:
    integral: float = 0.0


def _pi_step(
    error: float, state: _PIState, kp: float, ki: float, i_max: float
) -> float:
    """PI controller with integral anti-windup clamp.

    *error* must be non-negative (caller selects sign convention).
    Returns raw (unclamped) output.
    """
    state.integral = min(state.integral + error, i_max)
    return kp * error + ki * state.integral


# -- per-zone state --------------------------------------------------------

_WARMUP_JUMP_C = 3.0


@dataclass
class _ZoneState:
    fan_idx: int
    sat_idx: int
    temp_obs_idx: int
    fan_max: float  # design-max kg/s (from action space upper bound)
    air_pi: _PIState = field(default_factory=_PIState)
    sat_sp: float = 14.0  # current SAT setpoint [C], evolved by T&R
    prev_temp: float | None = None


# -- policy ----------------------------------------------------------------


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
    """G36-inspired PI airflow + Trim-and-Respond SAT controller for PSZ."""

    def __init__(self, policy_cfg: Any) -> None:
        self.heating_sp_c: float = float(
            getattr(policy_cfg, "heating_setpoint_c")
        )
        self.cooling_sp_c: float = float(
            getattr(policy_cfg, "cooling_setpoint_c")
        )

        # PI gains for zone-temp -> airflow loop
        self.kp: float = float(getattr(policy_cfg, "kp"))
        self.ki: float = float(getattr(policy_cfg, "ki"))
        self.integral_max: float = float(
            getattr(policy_cfg, "integral_max")
        )

        # Fan flow limits (fraction of design max)
        self.min_fan_frac: float = float(
            getattr(policy_cfg, "min_fan_fraction")
        )

        # SAT Trim-and-Respond parameters
        self.sat_min_c: float = float(getattr(policy_cfg, "sat_min_c"))
        self.sat_max_c: float = float(getattr(policy_cfg, "sat_max_c"))
        self.sat_initial_c: float = float(
            getattr(policy_cfg, "sat_initial_c")
        )
        self.sat_trim: float = float(getattr(policy_cfg, "sat_trim"))
        self.sat_respond: float = float(getattr(policy_cfg, "sat_respond"))
        self.demand_deadband: float = float(
            getattr(policy_cfg, "demand_deadband")
        )

        self.availability_on: float = float(
            getattr(policy_cfg, "availability_on")
        )

        sched_cfg = getattr(policy_cfg, "target_schedule", None)
        self._sched_enabled: bool = (
            bool(getattr(sched_cfg, "enabled"))
            if sched_cfg is not None
            else False
        )
        if self._sched_enabled and sched_cfg is not None:
            wkd = getattr(sched_cfg, "weekend_days")
            self._weekend_days: set[int] = (
                {int(x) for x in wkd} if isinstance(wkd, (list, tuple)) else {1, 7}
            )
            self._weekend_c: float = float(getattr(sched_cfg, "weekend_target_c"))
            self._weekday_c: float = float(getattr(sched_cfg, "weekday_target_c"))
            self._setback_c: float = float(getattr(sched_cfg, "weekday_setback_target_c"))
            self._setback_start: float = float(getattr(sched_cfg, "weekday_setback_start_hour"))
            self._setback_end: float = float(getattr(sched_cfg, "weekday_setback_end_hour"))
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

    # -- env binding -------------------------------------------------------

    def bind_env(self, env: Any) -> None:
        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")

        highs = None
        try:
            if hasattr(env, "action_space") and hasattr(env.action_space, "low"):
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

            temp_idx = find_zone_air_temp_index_for_zone(
                obs_names, zone_name=sys.zone
            )
            fan_max = float(highs[fan_idx]) if highs is not None else 1.0

            self._zones.append(
                _ZoneState(
                    fan_idx=fan_idx,
                    sat_idx=sat_idx,
                    temp_obs_idx=temp_idx,
                    fan_max=fan_max,
                    sat_sp=self.sat_initial_c,
                )
            )

        if not self._zones:
            logger.warning(
                "UnitaryG36Policy: no unitary systems discovered"
                " -- policy is a no-op"
            )

        if self._sched_enabled:
            try:
                self._tod_idx = find_time_of_day_index(obs_names)
                self._dow_idx = find_day_of_week_index(obs_names)
            except Exception:
                self._tod_idx = None
                self._dow_idx = None

        self._n_act = len(act_names)
        self.reset()

    # -- reset / schedule --------------------------------------------------

    def reset(self) -> None:
        for z in self._zones:
            z.air_pi = _PIState()
            z.sat_sp = self.sat_initial_c
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

    # -- control logic -----------------------------------------------------

    def _airflow_command(
        self, z: _ZoneState, t_zone: float, heat_sp: float, cool_sp: float
    ) -> float:
        """PI loop: zone temperature error -> fan mass flow rate [kg/s].

        In the deadband the integrator decays toward zero and the fan
        holds minimum flow, matching G36 minimum-ventilation behaviour.
        """
        m_dot_min = self.min_fan_frac * z.fan_max

        if t_zone > cool_sp:
            err = t_zone - cool_sp
        elif t_zone < heat_sp:
            err = heat_sp - t_zone
        else:
            z.air_pi.integral *= 0.8
            return m_dot_min

        u = _pi_step(err, z.air_pi, self.kp, self.ki, self.integral_max)
        m_dot = m_dot_min + u * (z.fan_max - m_dot_min)
        return float(np.clip(m_dot, m_dot_min, z.fan_max))

    def _sat_trim_and_respond(
        self, z: _ZoneState, t_zone: float, heat_sp: float, cool_sp: float
    ) -> float:
        """Trim-and-Respond SAT reset (heating + cooling).

        Zone too warm  -> respond down (lower SAT, more cooling).
        Zone too cold  -> respond up   (raise SAT, more heating).
        Zone satisfied -> trim toward sat_initial_c (neutral).
        """
        if t_zone - cool_sp > self.demand_deadband:
            z.sat_sp -= self.sat_respond
        elif heat_sp - t_zone > self.demand_deadband:
            z.sat_sp += self.sat_respond
        elif z.sat_sp < self.sat_initial_c:
            z.sat_sp = min(z.sat_sp + self.sat_trim, self.sat_initial_c)
        elif z.sat_sp > self.sat_initial_c:
            z.sat_sp = max(z.sat_sp - self.sat_trim, self.sat_initial_c)
        z.sat_sp = float(np.clip(z.sat_sp, self.sat_min_c, self.sat_max_c))
        return z.sat_sp

    # -- predict -----------------------------------------------------------

    def predict(
        self, obs: Any, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        if not self._zones:
            raise RuntimeError(
                "Policy not bound to an env; call bind_env() first."
            )

        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        heat_sp, cool_sp = self._current_setpoints(obs_arr)

        action = np.zeros(self._n_act, dtype=float)

        for idx in self._avail_idxs:
            action[idx] = self.availability_on

        for z in self._zones:
            tz = float(obs_arr[z.temp_obs_idx])

            if z.prev_temp is not None and abs(tz - z.prev_temp) > _WARMUP_JUMP_C:
                z.air_pi = _PIState()
            z.prev_temp = tz

            action[z.fan_idx] = self._airflow_command(z, tz, heat_sp, cool_sp)
            action[z.sat_idx] = self._sat_trim_and_respond(z, tz, heat_sp, cool_sp)

        return action, None

    # -- metrics -----------------------------------------------------------

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
