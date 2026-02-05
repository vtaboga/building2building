from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from b2b.baselines.common import (
    find_controlled_zone_air_temp_index,
    find_day_of_week_index,
    find_time_of_day_index,
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


class UnitaryAirflowFirstSatPolicy:
    """
    Adapter that exposes the SB3-like `predict()` API required by the benchmark.

    This reuses the same control logic as `b2b.baselines.runner` for
    `policy.type == "unitary_airflow_first_sat"`.
    """

    def __init__(self, policy_cfg: Any):
        # Core parameters
        self.target_temp_c = float(getattr(policy_cfg, "target_temp_c", 21.0))
        self.deadband_c = float(getattr(policy_cfg, "deadband_c", 0.5))

        # Availability override
        self.availability_on = float(getattr(policy_cfg, "availability_on", 2.0))

        # Fixed initial SAT setpoints on mode entry
        self.outlet_temp_heating_c = float(getattr(policy_cfg, "outlet_temp_heating_c", 35.0))
        self.outlet_temp_cooling_c = float(getattr(policy_cfg, "outlet_temp_cooling_c", 14.0))

        # PI on fan mass flow
        self.fan_base_kg_s = float(getattr(policy_cfg, "fan_base_kg_s", 0.3))
        self.kp = float(getattr(policy_cfg, "kp", 0.2))
        self.ki = float(getattr(policy_cfg, "ki", 0.0002))
        self.integral_limit = float(getattr(policy_cfg, "integral_limit", 200.0))
        self.fan_min_cfg = getattr(policy_cfg, "fan_min_kg_s", None)
        self.fan_max_cfg = getattr(policy_cfg, "fan_max_kg_s", None)

        # SAT trim/reset loop
        self.sat_min_c = float(getattr(policy_cfg, "sat_min_c", 10.0))
        self.sat_max_c = float(getattr(policy_cfg, "sat_max_c", 45.0))
        self.sat_step_c = float(getattr(policy_cfg, "sat_step_c", 0.2))
        self.sat_rate_limit_c_per_step = float(
            getattr(policy_cfg, "sat_rate_limit_c_per_step", 0.2)
        )
        self.sat_saturation_steps = int(getattr(policy_cfg, "sat_saturation_steps", 4))

        # Optional target schedule
        sched_cfg = getattr(policy_cfg, "target_schedule", None)
        enabled = bool(getattr(sched_cfg, "enabled", False)) if sched_cfg is not None else False
        weekend_days_raw = getattr(sched_cfg, "weekend_days", [1, 7]) if sched_cfg is not None else [1, 7]
        weekend_days = {int(x) for x in weekend_days_raw} if isinstance(weekend_days_raw, (list, tuple)) else {1, 7}
        self.target_schedule = TargetSchedule(
            enabled=enabled,
            weekend_days=weekend_days,
            weekend_target_c=float(getattr(sched_cfg, "weekend_target_c", self.target_temp_c))
            if sched_cfg is not None
            else self.target_temp_c,
            weekday_target_c=float(getattr(sched_cfg, "weekday_target_c", self.target_temp_c))
            if sched_cfg is not None
            else self.target_temp_c,
            weekday_setback_target_c=float(getattr(sched_cfg, "weekday_setback_target_c", self.target_temp_c))
            if sched_cfg is not None
            else self.target_temp_c,
            weekday_setback_start_hour=float(getattr(sched_cfg, "weekday_setback_start_hour", 9.0))
            if sched_cfg is not None
            else 9.0,
            weekday_setback_end_hour=float(getattr(sched_cfg, "weekday_setback_end_hour", 16.0))
            if sched_cfg is not None
            else 16.0,
        )

        # Populated by bind_env()
        self._idxs = None
        self._temp_idx: int | None = None
        self._tod_idx: int | None = None
        self._dow_idx: int | None = None
        self._fan_low: float = float("-inf")
        self._fan_high: float = float("inf")

        # Stateful controller bits
        self._pi_state = PIState(integral=0.0)
        self._sat_state = AirflowFirstSatState(sat_sp_c=float(self.target_temp_c))

    def bind_env(self, env: Any) -> None:
        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")
        controlled_zones = None
        if hasattr(env, "metadata") and isinstance(env.metadata, dict):
            cz = env.metadata.get("controlled_zones")
            if isinstance(cz, list) and all(isinstance(x, str) for x in cz):
                controlled_zones = cz

        self._idxs = select_unitary_actuator_indices(act_names)
        self._temp_idx = find_controlled_zone_air_temp_index(
            obs_names, controlled_zones=controlled_zones
        )

        if self.target_schedule.enabled:
            try:
                self._tod_idx = find_time_of_day_index(obs_names)
                self._dow_idx = find_day_of_week_index(obs_names)
            except Exception:
                self._tod_idx = None
                self._dow_idx = None

        # Fan bounds from action space (preferred)
        self._fan_low = float("-inf")
        self._fan_high = float("inf")
        try:
            if hasattr(env, "action_space") and hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                lows = np.asarray(env.action_space.low, dtype=float).reshape(-1)
                highs = np.asarray(env.action_space.high, dtype=float).reshape(-1)
                i0 = int(self._idxs.idx_fans[0])
                if 0 <= i0 < len(lows) and 0 <= i0 < len(highs):
                    self._fan_low = float(lows[i0])
                    self._fan_high = float(highs[i0])
        except Exception:
            pass

        if isinstance(self.fan_min_cfg, (int, float)):
            self._fan_low = max(self._fan_low, float(self.fan_min_cfg))
        if isinstance(self.fan_max_cfg, (int, float)):
            self._fan_high = min(self._fan_high, float(self.fan_max_cfg))

        self.reset()

    def reset(self) -> None:
        self._pi_state = PIState(integral=0.0)
        self._sat_state = AirflowFirstSatState(sat_sp_c=float(self.target_temp_c))

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
        if self._idxs is None or self._temp_idx is None:
            raise RuntimeError("Policy is not bound to an env; call bind_env(env) first.")

        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        tz = float(obs_arr[int(self._temp_idx)]) if int(self._temp_idx) < len(obs_arr) else float("nan")
        target_now = float(self._scheduled_target_c(obs_arr=obs_arr))
        deadband = float(self.deadband_c)

        if tz < target_now - deadband:
            mode = "heating"
        elif tz > target_now + deadband:
            mode = "cooling"
        else:
            mode = "deadband"

        fan_cmd = compute_fan_command_pi(
            state=self._pi_state,
            tz_c=tz,
            target_c=target_now,
            deadband_c=deadband,
            fan_base_kg_s=float(self.fan_base_kg_s),
            kp=float(self.kp),
            ki=float(self.ki),
            integral_limit=float(self.integral_limit),
            mode=mode,  # type: ignore[arg-type]
        )

        fan_cmd_for_logic = float(np.clip(float(fan_cmd), float(self._fan_low), float(self._fan_high)))

        # Most action spaces in our benchmarks are 2D (fan + one node setpoint),
        # but we keep this general for environments that also expose availability.
        n_act = max(
            max(self._idxs.idx_outlet_nodes, default=-1),
            max(self._idxs.idx_fans, default=-1),
            max(self._idxs.idx_avail, default=-1),
        ) + 1
        action_cmd = np.zeros((int(n_act),), dtype=float)

        for idx in self._idxs.idx_avail:
            action_cmd[int(idx)] = float(self.availability_on)

        for idx in self._idxs.idx_fans:
            action_cmd[int(idx)] = float(np.clip(float(fan_cmd), float(self._fan_low), float(self._fan_high)))

        outlet_sp = compute_outlet_sat_sp_airflow_first(
            state=self._sat_state,
            tz_c=tz,
            target_c=target_now,
            deadband_c=deadband,
            fan_cmd_kg_s=fan_cmd_for_logic,
            fan_min_kg_s=float(self._fan_low),
            fan_max_kg_s=float(self._fan_high),
            mode=mode,  # type: ignore[arg-type]
            outlet_sp_heating_c=float(self.outlet_temp_heating_c),
            outlet_sp_cooling_c=float(self.outlet_temp_cooling_c),
            sat_min_c=float(self.sat_min_c),
            sat_max_c=float(self.sat_max_c),
            sat_step_c=float(self.sat_step_c),
            sat_rate_limit_c_per_step=float(self.sat_rate_limit_c_per_step),
            sat_saturation_steps=int(self.sat_saturation_steps),
        )

        for idx in self._idxs.idx_outlet_nodes:
            action_cmd[int(idx)] = float(outlet_sp)

        return action_cmd, None

    def step_metrics(self, obs: Any, *, action: np.ndarray) -> dict[str, float]:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        target_now = float(self._scheduled_target_c(obs_arr=obs_arr))
        tz = float("nan")
        if self._temp_idx is not None and int(self._temp_idx) < len(obs_arr):
            tz = float(obs_arr[int(self._temp_idx)])
        return {
            "target_temp_c": float(target_now),
            "conditioned_zone_temp_c": float(tz),
        }

@dataclass
class AirflowFirstSatState:
    """
    Minimal state for a rule-based "airflow first, then SAT trim/reset" controller.

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
    """
    Secondary loop (SAT trim/reset) for the "airflow-first" sequence.

    Rules:
    - Use airflow as the fast loop.
    - Only adjust SAT when airflow is pegged at min/max for `sat_saturation_steps`
      *and* the zone is still outside the deadband in the same direction.
    - SAT changes are clamped and rate-limited per timestep.
    """
    if sat_saturation_steps <= 0:
        raise ValueError("sat_saturation_steps must be >= 1")
    if sat_rate_limit_c_per_step < 0.0:
        raise ValueError("sat_rate_limit_c_per_step must be >= 0")
    if sat_step_c < 0.0:
        raise ValueError("sat_step_c must be >= 0")

    # Mode transitions: re-initialize SAT to a reasonable fixed value and
    # clear saturation timing.
    if mode != state.last_mode:
        state.at_limit_steps = 0
        state.last_mode = mode
        if mode == "heating":
            state.sat_sp_c = float(outlet_sp_heating_c)
        elif mode == "cooling":
            state.sat_sp_c = float(outlet_sp_cooling_c)
        else:
            state.sat_sp_c = float(target_c)

    # In deadband, don't trim SAT beyond keeping it near the comfort target.
    if mode == "deadband":
        state.at_limit_steps = 0
        state.sat_sp_c = float(target_c)
        return float(state.sat_sp_c)

    # Determine if we are still in error for this mode.
    if mode == "heating":
        in_error = tz_c < (target_c - deadband_c)
        # Heating: if airflow maxed but still cold -> raise SAT; if airflow min and
        # still cold -> also raise SAT (but airflow-first means we only adjust SAT
        # at limits; both min/max are "limits" but max is the typical heating limit).
        sat_direction = +1.0
    else:  # mode == "cooling"
        in_error = tz_c > (target_c + deadband_c)
        # Cooling: if airflow maxed but still warm -> lower SAT; if airflow min and
        # still warm -> also lower SAT (again, max is typical).
        sat_direction = -1.0

    at_fan_max = fan_cmd_kg_s >= (fan_max_kg_s - fan_limit_epsilon_kg_s)
    at_fan_min = fan_cmd_kg_s <= (fan_min_kg_s + fan_limit_epsilon_kg_s)
    at_limit = bool(at_fan_max or at_fan_min)

    if at_limit and in_error:
        state.at_limit_steps += 1
    else:
        state.at_limit_steps = 0

    if state.at_limit_steps >= sat_saturation_steps:
        # Apply a discrete trim step, then reset the timer so we only move
        # SAT every `sat_saturation_steps` while saturated.
        desired = float(state.sat_sp_c + sat_direction * sat_step_c)
        desired = float(max(sat_min_c, min(sat_max_c, desired)))

        # Per-step rate limit (a no-op when large).
        delta = desired - float(state.sat_sp_c)
        if sat_rate_limit_c_per_step > 0.0:
            delta = float(
                max(-sat_rate_limit_c_per_step, min(sat_rate_limit_c_per_step, delta))
            )
        state.sat_sp_c = float(state.sat_sp_c + delta)
        state.at_limit_steps = 0

    # Always enforce SAT clamp (defensive).
    state.sat_sp_c = float(max(sat_min_c, min(sat_max_c, float(state.sat_sp_c))))
    return float(state.sat_sp_c)

