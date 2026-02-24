"""
Baseline controller for unitary HVAC systems (PSZ, split systems, etc.).

Supports buildings with one or many independent unitary systems, each
serving a single zone.  Each system gets its own PI + SAT-trim state
so that zones are controlled independently.

Per system the action space is:
    - 1 fan mass flow rate
    - 1 outlet temperature setpoint

Control strategy (per system):
    1. Primary loop: PI on fan mass flow driven by zone temperature error.
    2. Secondary loop: SAT trim/reset — only adjusts outlet temperature
       when fan is saturated for several consecutive steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from b2b.baselines.common import (
    find_day_of_week_index,
    find_time_of_day_index,
    find_zone_air_temp_index_for_zone,
    require_env_metadata_list_str,
)
from b2b.baselines.controllers.unitary_pi import PIState, compute_fan_command_pi
from b2b.baselines.unitary_actuators import select_unitary_actuator_indices


UnitaryAirflowFirstMode = Literal["heating", "cooling", "deadband"]


@dataclass(frozen=True, slots=True)
class TargetSchedule:
    enabled: bool
    weekend_days: set[int]
    weekend_target_c: float
    weekday_target_c: float
    weekday_setback_target_c: float
    weekday_setback_start_hour: float
    weekday_setback_end_hour: float


@dataclass
class _SystemState:
    """Per-unitary-system mutable control state."""

    fan_idx: int
    outlet_idx: int
    temp_obs_idx: int
    fan_low: float
    fan_high: float
    pi: PIState
    sat: AirflowFirstSatState


def _match_actuator_index(act_names: list[str], component_type: str, control_type: str, component_name: str) -> int:
    target = f"{component_type}::{control_type}::{component_name}"
    for i, name in enumerate(act_names):
        if name == target:
            return i
    raise RuntimeError(f"Could not find action for actuator: {target!r}")


class UnitaryAirflowFirstSatPolicy:
    """
    Multi-system unitary baseline with SB3-like ``predict()`` API.

    Discovers per-system (fan, outlet, zone) groupings from
    ``env.metadata["hvac_equipment"]`` when available, otherwise falls back
    to the flat ``select_unitary_actuator_indices`` heuristic (legacy
    single-system behaviour).
    """

    def __init__(self, policy_cfg: Any):
        self.target_temp_c = float(getattr(policy_cfg, "target_temp_c", 21.0))
        self.deadband_c = float(getattr(policy_cfg, "deadband_c", 0.5))

        self.availability_on = float(getattr(policy_cfg, "availability_on", 2.0))

        self.outlet_temp_heating_c = float(getattr(policy_cfg, "outlet_temp_heating_c", 35.0))
        self.outlet_temp_cooling_c = float(getattr(policy_cfg, "outlet_temp_cooling_c", 14.0))

        self.fan_base_kg_s = float(getattr(policy_cfg, "fan_base_kg_s", 0.3))
        self.kp = float(getattr(policy_cfg, "kp", 0.2))
        self.ki = float(getattr(policy_cfg, "ki", 0.0002))
        self.integral_limit = float(getattr(policy_cfg, "integral_limit", 200.0))
        self.fan_min_cfg = getattr(policy_cfg, "fan_min_kg_s", None)
        self.fan_max_cfg = getattr(policy_cfg, "fan_max_kg_s", None)

        self.sat_min_c = float(getattr(policy_cfg, "sat_min_c", 10.0))
        self.sat_max_c = float(getattr(policy_cfg, "sat_max_c", 45.0))
        self.sat_step_c = float(getattr(policy_cfg, "sat_step_c", 0.2))
        self.sat_rate_limit_c_per_step = float(
            getattr(policy_cfg, "sat_rate_limit_c_per_step", 0.2)
        )
        self.sat_saturation_steps = int(getattr(policy_cfg, "sat_saturation_steps", 4))

        sched_cfg = getattr(policy_cfg, "target_schedule", None)
        enabled = bool(getattr(sched_cfg, "enabled", False)) if sched_cfg is not None else False
        weekend_days_raw = getattr(sched_cfg, "weekend_days", [1, 7]) if sched_cfg is not None else [1, 7]
        weekend_days = {int(x) for x in weekend_days_raw} if isinstance(weekend_days_raw, (list, tuple)) else {1, 7}
        self.target_schedule = TargetSchedule(
            enabled=enabled,
            weekend_days=weekend_days,
            weekend_target_c=float(getattr(sched_cfg, "weekend_target_c", self.target_temp_c))
            if sched_cfg is not None else self.target_temp_c,
            weekday_target_c=float(getattr(sched_cfg, "weekday_target_c", self.target_temp_c))
            if sched_cfg is not None else self.target_temp_c,
            weekday_setback_target_c=float(getattr(sched_cfg, "weekday_setback_target_c", self.target_temp_c))
            if sched_cfg is not None else self.target_temp_c,
            weekday_setback_start_hour=float(getattr(sched_cfg, "weekday_setback_start_hour", 9.0))
            if sched_cfg is not None else 9.0,
            weekday_setback_end_hour=float(getattr(sched_cfg, "weekday_setback_end_hour", 16.0))
            if sched_cfg is not None else 16.0,
        )

        self._systems: list[_SystemState] = []
        self._avail_idxs: list[int] = []
        self._n_act: int = 0
        self._tod_idx: int | None = None
        self._dow_idx: int | None = None

    # ------------------------------------------------------------------
    # Environment binding
    # ------------------------------------------------------------------

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

        if unitary_systems:
            self._bind_from_equipment(unitary_systems, obs_names, act_names, lows, highs)
        else:
            self._bind_legacy(obs_names, act_names, env)

        if self.target_schedule.enabled:
            try:
                self._tod_idx = find_time_of_day_index(obs_names)
                self._dow_idx = find_day_of_week_index(obs_names)
            except Exception:
                self._tod_idx = None
                self._dow_idx = None

        self._n_act = len(act_names)
        self.reset()

    def _bind_from_equipment(
        self,
        unitary_systems: list[Any],
        obs_names: list[str],
        act_names: list[str],
        lows: np.ndarray | None,
        highs: np.ndarray | None,
    ) -> None:
        """Build per-system state from ``UnitarySystem`` equipment objects."""
        self._avail_idxs = []
        self._systems = []
        seen_avail: set[int] = set()

        for sys in unitary_systems:
            fan_idx: int | None = None
            outlet_idx: int | None = None
            for act in sys.actuator_descriptions():
                idx = _match_actuator_index(
                    act_names, act.component_type, act.control_type, act.component_name
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
                    outlet_idx = idx

            if fan_idx is None or outlet_idx is None:
                continue

            temp_idx = find_zone_air_temp_index_for_zone(obs_names, zone_name=sys.zone)

            fan_low, fan_high = self._resolve_fan_bounds(fan_idx, lows, highs)
            self._systems.append(_SystemState(
                fan_idx=fan_idx,
                outlet_idx=outlet_idx,
                temp_obs_idx=temp_idx,
                fan_low=fan_low,
                fan_high=fan_high,
                pi=PIState(integral=0.0),
                sat=AirflowFirstSatState(sat_sp_c=float(self.target_temp_c)),
            ))

    def _bind_legacy(
        self,
        obs_names: list[str],
        act_names: list[str],
        env: Any,
    ) -> None:
        """Fallback: pair fans and outlets positionally (single-system compat)."""
        from b2b.baselines.common import find_controlled_zone_air_temp_index

        idxs = select_unitary_actuator_indices(act_names)
        self._avail_idxs = list(idxs.idx_avail)

        controlled_zones = None
        if hasattr(env, "metadata") and isinstance(env.metadata, dict):
            cz = env.metadata.get("controlled_zones")
            if isinstance(cz, list) and all(isinstance(x, str) for x in cz):
                controlled_zones = cz

        temp_idx = find_controlled_zone_air_temp_index(obs_names, controlled_zones=controlled_zones)

        lows = highs = None
        try:
            if hasattr(env, "action_space") and hasattr(env.action_space, "low"):
                lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
                highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
        except Exception:
            pass

        n_systems = min(len(idxs.idx_fans), len(idxs.idx_outlet_nodes))
        self._systems = []
        for k in range(n_systems):
            fi = idxs.idx_fans[k]
            oi = idxs.idx_outlet_nodes[k]
            fan_low, fan_high = self._resolve_fan_bounds(fi, lows, highs)
            self._systems.append(_SystemState(
                fan_idx=fi,
                outlet_idx=oi,
                temp_obs_idx=temp_idx,
                fan_low=fan_low,
                fan_high=fan_high,
                pi=PIState(integral=0.0),
                sat=AirflowFirstSatState(sat_sp_c=float(self.target_temp_c)),
            ))

    def _resolve_fan_bounds(
        self,
        fan_idx: int,
        lows: np.ndarray | None,
        highs: np.ndarray | None,
    ) -> tuple[float, float]:
        fan_low = float("-inf")
        fan_high = float("inf")
        if lows is not None and highs is not None:
            if 0 <= fan_idx < len(lows):
                fan_low = float(lows[fan_idx])
            if 0 <= fan_idx < len(highs):
                fan_high = float(highs[fan_idx])
        if isinstance(self.fan_min_cfg, (int, float)):
            fan_low = max(fan_low, float(self.fan_min_cfg))
        if isinstance(self.fan_max_cfg, (int, float)):
            fan_high = min(fan_high, float(self.fan_max_cfg))
        return fan_low, fan_high

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        for s in self._systems:
            s.pi = PIState(integral=0.0)
            s.sat = AirflowFirstSatState(sat_sp_c=float(self.target_temp_c))

    def _scheduled_target_c(self, *, obs_arr: np.ndarray) -> float:
        if not self.target_schedule.enabled:
            return float(self.target_temp_c)
        if self._tod_idx is None or self._dow_idx is None:
            return float(self.target_temp_c)
        if self._tod_idx >= len(obs_arr) or self._dow_idx >= len(obs_arr):
            return float(self.target_temp_c)

        hour = float(obs_arr[int(self._tod_idx)])
        day = int(obs_arr[int(self._dow_idx)])

        if day in self.target_schedule.weekend_days:
            return float(self.target_schedule.weekend_target_c)

        in_setback = (
            float(self.target_schedule.weekday_setback_start_hour)
            <= hour
            < float(self.target_schedule.weekday_setback_end_hour)
        )
        if in_setback:
            return float(self.target_schedule.weekday_setback_target_c)
        return float(self.target_schedule.weekday_target_c)

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        if not self._systems:
            raise RuntimeError("Policy is not bound to an env; call bind_env(env) first.")

        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        target_now = float(self._scheduled_target_c(obs_arr=obs_arr))
        deadband = float(self.deadband_c)

        action_cmd = np.zeros(self._n_act, dtype=float)

        for idx in self._avail_idxs:
            action_cmd[int(idx)] = float(self.availability_on)

        for sys in self._systems:
            tz = (
                float(obs_arr[sys.temp_obs_idx])
                if sys.temp_obs_idx < len(obs_arr)
                else float("nan")
            )

            if tz < target_now - deadband:
                mode: UnitaryAirflowFirstMode = "heating"
            elif tz > target_now + deadband:
                mode = "cooling"
            else:
                mode = "deadband"

            fan_cmd = compute_fan_command_pi(
                state=sys.pi,
                tz_c=tz,
                target_c=target_now,
                deadband_c=deadband,
                fan_base_kg_s=float(self.fan_base_kg_s),
                kp=float(self.kp),
                ki=float(self.ki),
                integral_limit=float(self.integral_limit),
                mode=mode,
            )
            fan_clipped = float(np.clip(fan_cmd, sys.fan_low, sys.fan_high))
            action_cmd[sys.fan_idx] = fan_clipped

            outlet_sp = compute_outlet_sat_sp_airflow_first(
                state=sys.sat,
                tz_c=tz,
                target_c=target_now,
                deadband_c=deadband,
                fan_cmd_kg_s=fan_clipped,
                fan_min_kg_s=sys.fan_low,
                fan_max_kg_s=sys.fan_high,
                mode=mode,
                outlet_sp_heating_c=float(self.outlet_temp_heating_c),
                outlet_sp_cooling_c=float(self.outlet_temp_cooling_c),
                sat_min_c=float(self.sat_min_c),
                sat_max_c=float(self.sat_max_c),
                sat_step_c=float(self.sat_step_c),
                sat_rate_limit_c_per_step=float(self.sat_rate_limit_c_per_step),
                sat_saturation_steps=int(self.sat_saturation_steps),
            )
            action_cmd[sys.outlet_idx] = outlet_sp

        return action_cmd, None

    def step_metrics(self, obs: Any, *, action: np.ndarray) -> dict[str, float]:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        target_now = float(self._scheduled_target_c(obs_arr=obs_arr))
        temps: list[float] = []
        for sys in self._systems:
            if sys.temp_obs_idx < len(obs_arr):
                temps.append(float(obs_arr[sys.temp_obs_idx]))
        metrics: dict[str, float] = {"target_temp_c": target_now}
        if temps:
            metrics["mean_zone_temp_c"] = float(np.mean(temps))
            metrics["min_zone_temp_c"] = float(np.min(temps))
            metrics["max_zone_temp_c"] = float(np.max(temps))
        return metrics


# ======================================================================
# Pure SAT trim/reset logic (unchanged, shared with other modules)
# ======================================================================


@dataclass
class AirflowFirstSatState:
    """Minimal state for a rule-based "airflow first, then SAT trim/reset" controller.

    - sat_sp_c: current outlet (supply) air temperature setpoint [°C]
    - at_limit_steps: consecutive steps with airflow at a limit while still in error
    - last_mode: last inferred operating mode for integrator/state resets
    """

    sat_sp_c: float
    at_limit_steps: int = 0
    last_mode: UnitaryAirflowFirstMode = "deadband"


def compute_outlet_sat_sp_airflow_first(
    *,
    state: AirflowFirstSatState,
    tz_c: float,
    target_c: float,
    deadband_c: float,
    fan_cmd_kg_s: float,
    fan_min_kg_s: float,
    fan_max_kg_s: float,
    mode: UnitaryAirflowFirstMode,
    outlet_sp_heating_c: float,
    outlet_sp_cooling_c: float,
    sat_min_c: float,
    sat_max_c: float,
    sat_step_c: float,
    sat_rate_limit_c_per_step: float,
    sat_saturation_steps: int,
    fan_limit_epsilon_kg_s: float = 1e-6,
) -> float:
    """Secondary loop (SAT trim/reset) for the "airflow-first" sequence.

    Rules:
    - Use airflow as the fast loop.
    - Only adjust SAT when airflow is pegged at min/max for ``sat_saturation_steps``
      *and* the zone is still outside the deadband in the same direction.
    - SAT changes are clamped and rate-limited per timestep.
    """
    if sat_saturation_steps <= 0:
        raise ValueError("sat_saturation_steps must be >= 1")
    if sat_rate_limit_c_per_step < 0.0:
        raise ValueError("sat_rate_limit_c_per_step must be >= 0")
    if sat_step_c < 0.0:
        raise ValueError("sat_step_c must be >= 0")

    if mode != state.last_mode:
        state.at_limit_steps = 0
        state.last_mode = mode
        if mode == "heating":
            state.sat_sp_c = float(outlet_sp_heating_c)
        elif mode == "cooling":
            state.sat_sp_c = float(outlet_sp_cooling_c)
        else:
            state.sat_sp_c = float(target_c)

    if mode == "deadband":
        state.at_limit_steps = 0
        state.sat_sp_c = float(target_c)
        return float(state.sat_sp_c)

    if mode == "heating":
        in_error = tz_c < (target_c - deadband_c)
        sat_direction = +1.0
    else:
        in_error = tz_c > (target_c + deadband_c)
        sat_direction = -1.0

    at_fan_max = fan_cmd_kg_s >= (fan_max_kg_s - fan_limit_epsilon_kg_s)
    at_fan_min = fan_cmd_kg_s <= (fan_min_kg_s + fan_limit_epsilon_kg_s)
    at_limit = bool(at_fan_max or at_fan_min)

    if at_limit and in_error:
        state.at_limit_steps += 1
    else:
        state.at_limit_steps = 0

    if state.at_limit_steps >= sat_saturation_steps:
        desired = float(state.sat_sp_c + sat_direction * sat_step_c)
        desired = float(max(sat_min_c, min(sat_max_c, desired)))

        delta = desired - float(state.sat_sp_c)
        if sat_rate_limit_c_per_step > 0.0:
            delta = float(
                max(-sat_rate_limit_c_per_step, min(sat_rate_limit_c_per_step, delta))
            )
        state.sat_sp_c = float(state.sat_sp_c + delta)
        state.at_limit_steps = 0

    state.sat_sp_c = float(max(sat_min_c, min(sat_max_c, float(state.sat_sp_c))))
    return float(state.sat_sp_c)
