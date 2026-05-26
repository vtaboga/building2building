"""Controller for single-zone unitary HVAC systems.

Each conditioned zone has an independent Packaged Single Zone (PSZ) with two
actuators:
  - Fan Air Mass Flow Rate [kg/s]
  - Supply Air Temperature setpoint [C]

Control strategy:
  1. Zone temperature PI loop modulates fan airflow as a capacity proxy.
  2. Trim-and-Respond SAT reset adjusts supply temperature based on zone
     demand: zone too warm -> respond down (lower SAT), zone too cold ->
     respond up (raise SAT), zone satisfied -> trim toward neutral.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

FanErrorMode = Literal["nearest_setpoint", "center_of_band"]
_FAN_ERROR_MODES: tuple[FanErrorMode, ...] = ("nearest_setpoint", "center_of_band")

from baselines.utils.metadata import (
    find_obs_index,
    find_obs_index_optional,
    find_zone_air_temp_index,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass config (replaces OmegaConf)
# ---------------------------------------------------------------------------


@dataclass
class TargetScheduleConfig:
    enabled: bool = False
    weekend_days: tuple[int, ...] = (1, 7)
    weekend_target_c: float = 18.0
    weekday_target_c: float = 21.0
    weekday_setback_target_c: float = 18.0
    weekday_setback_start_hour: float = 9.0
    weekday_setback_end_hour: float = 16.0


@dataclass
class UnitaryHvacConfig:
    heating_setpoint_c: float = 20.0
    cooling_setpoint_c: float = 22.0
    kp: float = 0.8
    ki: float = 0.02
    integral_max: float = 5.0
    min_fan_fraction: float = 0.3
    sat_min_c: float = 12.0
    sat_max_c: float = 40.0
    sat_initial_c: float = 14.0
    sat_trim: float = 0.2
    sat_respond: float = 0.5
    demand_deadband: float = 0.5
    availability_on: float = 1.0
    # How the fan-airflow PI loop computes its error signal.
    # - "nearest_setpoint": err = max(0, t_zone - cool_sp) when hot,
    #                       max(0, heat_sp - t_zone) when cold,
    #                       0 in between (fan pinned at min inside the band).
    # - "center_of_band":  err = |t_zone - (heat_sp + cool_sp) / 2|, the PI
    #                       loop is active for every non-zero deviation from
    #                       the band center.
    fan_error_mode: FanErrorMode = "nearest_setpoint"
    target_schedule: TargetScheduleConfig = field(default_factory=TargetScheduleConfig)

    def __post_init__(self) -> None:
        if self.fan_error_mode not in _FAN_ERROR_MODES:
            raise ValueError(
                f"fan_error_mode must be one of {_FAN_ERROR_MODES}; "
                f"got {self.fan_error_mode!r}."
            )


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _PIState:
    integral: float = 0.0


def _pi_step(
    error: float, state: _PIState, kp: float, ki: float, i_max: float
) -> float:
    state.integral = min(state.integral + error, i_max)
    return kp * error + ki * state.integral


_WARMUP_JUMP_C = 3.0


@dataclass
class _ZoneState:
    fan_idx: int
    sat_idx: int
    temp_obs_idx: int
    fan_max: float
    target_obs_idx: int | None = None
    air_pi: _PIState = field(default_factory=_PIState)
    sat_sp: float = 14.0
    prev_temp: float | None = None


@dataclass
class _BaseboardState:
    htg_sp_idx: int
    temp_obs_idx: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _find_target_temp_index(obs_names: list[str], zone_name: str) -> int | None:
    prefix = "target_temperature"
    zn = zone_name.strip().lower()
    for i, name in enumerate(obs_names):
        nl = name.strip().lower()
        if nl.startswith(prefix):
            zone_part = nl[len(prefix) :].strip()
            if zone_part == zn or zn in zone_part or zone_part in zn:
                return i
    return None


def _require_metadata_list_str(env: Any, key: str) -> list[str]:
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("env.metadata missing")
    raw = env.metadata.get(key)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"env.metadata['{key}'] must be a list[str]")
    return raw


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class UnitaryHvacPolicy:
    """PI airflow + Trim-and-Respond SAT controller for PSZ.

    Usage::

        cfg = UnitaryHvacConfig(heating_setpoint_c=20, cooling_setpoint_c=22)
        policy = UnitaryHvacPolicy(cfg)
        policy.bind_env(env)
        obs, _ = env.reset()
        action, _ = policy.predict(obs)
    """

    def __init__(self, cfg: UnitaryHvacConfig | None = None) -> None:
        if cfg is None:
            cfg = UnitaryHvacConfig()

        self.heating_sp_c = cfg.heating_setpoint_c
        self.cooling_sp_c = cfg.cooling_setpoint_c
        if self.cooling_sp_c <= self.heating_sp_c:
            raise ValueError(
                f"cooling_setpoint_c ({self.cooling_sp_c}) must be > "
                f"heating_setpoint_c ({self.heating_sp_c})"
            )

        self.kp = cfg.kp
        self.ki = cfg.ki
        self.integral_max = cfg.integral_max
        self.min_fan_frac = cfg.min_fan_fraction

        self.sat_min_c = cfg.sat_min_c
        self.sat_max_c = cfg.sat_max_c
        self.sat_initial_c = cfg.sat_initial_c
        self.sat_trim = cfg.sat_trim
        self.sat_respond = cfg.sat_respond
        self.demand_deadband = cfg.demand_deadband
        self.availability_on = cfg.availability_on
        self.fan_error_mode: FanErrorMode = cfg.fan_error_mode

        sched = cfg.target_schedule
        self._sched_enabled = sched.enabled
        self._weekend_days: set[int] = set(sched.weekend_days)
        self._weekend_c = sched.weekend_target_c
        self._weekday_c = sched.weekday_target_c
        self._setback_c = sched.weekday_setback_target_c
        self._setback_start = sched.weekday_setback_start_hour
        self._setback_end = sched.weekday_setback_end_hour

        self._zones: list[_ZoneState] = []
        self._baseboards: list[_BaseboardState] = []
        self._avail_idxs: list[int] = []
        self._n_act: int = 0
        self._tod_idx: int | None = None
        self._dow_idx: int | None = None

    def bind_env(self, env: Any) -> None:
        """Wire observation/action indices from ``env.metadata``."""
        obs_names = _require_metadata_list_str(env, "observation_names")
        act_names = _require_metadata_list_str(env, "action_names")

        highs = None
        try:
            if hasattr(env, "action_space") and hasattr(env.action_space, "high"):
                highs = np.asarray(env.action_space.high, dtype=float).ravel()
        except Exception:
            pass

        equipment = (
            env.metadata.get("hvac_equipment", [])
            if hasattr(env, "metadata") and isinstance(env.metadata, dict)
            else []
        )
        unitary_systems = [
            e
            for e in equipment
            if getattr(e, "equipment_type", None) == "unitarysystem"
        ]

        self._avail_idxs = []
        self._zones = []
        seen_avail: set[int] = set()

        for sys in unitary_systems:
            fan_idx: int | None = None
            sat_idx: int | None = None
            for act in sys.actuator_descriptions():
                idx = _match_actuator_index(
                    act_names,
                    act.component_type,
                    act.control_type,
                    act.component_name,
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

            temp_idx = find_zone_air_temp_index(obs_names, sys.zones()[0])
            fan_max = float(highs[fan_idx]) if highs is not None else 1.0

            self._zones.append(
                _ZoneState(
                    fan_idx=fan_idx,
                    sat_idx=sat_idx,
                    temp_obs_idx=temp_idx,
                    fan_max=fan_max,
                    target_obs_idx=_find_target_temp_index(obs_names, sys.zones()[0]),
                    sat_sp=self.sat_initial_c,
                )
            )

        heating_only = [
            e for e in equipment if getattr(e, "equipment_type", None) == "heating_only"
        ]

        self._baseboards = []
        for bb in heating_only:
            for act in bb.actuator_descriptions():
                target = (
                    f"{act.component_type}::{act.control_type}"
                    f"::{act.component_name}"
                )
                if target not in act_names:
                    continue
                idx = _match_actuator_index(
                    act_names,
                    act.component_type,
                    act.control_type,
                    act.component_name,
                )
                temp_idx = find_zone_air_temp_index(obs_names, bb.zones()[0])
                self._baseboards.append(
                    _BaseboardState(htg_sp_idx=idx, temp_obs_idx=temp_idx)
                )

        if not self._zones and not self._baseboards:
            logger.warning(
                "UnitaryHvacPolicy: no unitary systems or heating-only zones "
                "discovered -- policy is a no-op"
            )

        if self._sched_enabled:
            self._tod_idx = find_obs_index_optional(obs_names, "time_of_day")
            self._dow_idx = find_obs_index_optional(obs_names, "day_of_week")

        self._n_act = len(act_names)
        self.reset()

    def reset(self) -> None:
        for z in self._zones:
            z.air_pi = _PIState()
            z.sat_sp = self.sat_initial_c
            z.prev_temp = None

    def _current_setpoints(
        self, obs_arr: np.ndarray, zone: _ZoneState | None = None
    ) -> tuple[float, float]:
        gap = self.cooling_sp_c - self.heating_sp_c

        if zone is not None and zone.target_obs_idx is not None:
            target = float(obs_arr[zone.target_obs_idx])
            half = gap / 2.0
            return target - half, target + half

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

        return sp, sp + gap

    def _airflow_command(
        self, z: _ZoneState, t_zone: float, heat_sp: float, cool_sp: float
    ) -> float:
        m_dot_min = self.min_fan_frac * z.fan_max
        if self.fan_error_mode == "center_of_band":
            center = 0.5 * (heat_sp + cool_sp)
            err = abs(t_zone - center)
            if err <= 0.0:
                z.air_pi.integral *= 0.8
                return m_dot_min
        else:
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

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        if not self._zones and not self._baseboards:
            raise RuntimeError("Policy not bound to an env; call bind_env() first.")

        obs_arr = np.asarray(obs, dtype=float).ravel()
        action = np.zeros(self._n_act, dtype=float)

        for idx in self._avail_idxs:
            action[idx] = self.availability_on

        for z in self._zones:
            heat_sp, cool_sp = self._current_setpoints(obs_arr, zone=z)
            tz = float(obs_arr[z.temp_obs_idx])

            if z.prev_temp is not None and abs(tz - z.prev_temp) > _WARMUP_JUMP_C:
                z.air_pi = _PIState()
            z.prev_temp = tz

            action[z.fan_idx] = self._airflow_command(z, tz, heat_sp, cool_sp)
            action[z.sat_idx] = self._sat_trim_and_respond(z, tz, heat_sp, cool_sp)

        for bb in self._baseboards:
            heat_sp, _ = self._current_setpoints(obs_arr)
            action[bb.htg_sp_idx] = heat_sp

        return action, None
