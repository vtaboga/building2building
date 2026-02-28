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
from b2b.baselines.unitary_actuators import select_unitary_actuator_indices


UnitaryPIMode = Literal["heating", "cooling", "deadband"]


@dataclass
class PIState:
    integral: float = 0.0
    last_mode: UnitaryPIMode = "deadband"


def compute_fan_command_pi(
    *,
    state: PIState,
    tz_c: float,
    target_c: float,
    deadband_c: float,
    fan_base_kg_s: float,
    kp: float,
    ki: float,
    integral_limit: float,
    mode: UnitaryPIMode,
) -> float:
    # Reset integrator when switching between heating/cooling/deadband.
    if mode != state.last_mode:
        state.integral = 0.0
        state.last_mode = mode

    # Mode-aware error sign so airflow increases in BOTH heating and cooling.
    #
    # - heating: tz below target => positive error
    # - cooling: tz above target => positive error
    # - deadband: zero error
    if mode == "deadband":
        err = 0.0
    elif mode == "heating":
        err = float(target_c - tz_c)
    else:  # mode == "cooling"
        err = float(tz_c - target_c)

    # Deadband: avoid chattering and integral windup near setpoint.
    if abs(err) <= deadband_c:
        err_pi = 0.0
    else:
        err_pi = err

    if err_pi != 0.0 and ki != 0.0:
        state.integral = float(
            max(-integral_limit, min(integral_limit, state.integral + err_pi))
        )

    u = float(fan_base_kg_s + kp * err_pi + ki * state.integral)
    return u


@dataclass(frozen=True, slots=True)
class TargetSchedule:
    enabled: bool
    weekend_days: set[int]
    weekend_target_c: float
    weekday_target_c: float
    weekday_setback_target_c: float
    weekday_setback_start_hour: float
    weekday_setback_end_hour: float


class UnitaryPIPolicy:
    """
    PI controller on fan airflow + fixed outlet node temperature setpoint.

    Exposes an SB3-like `predict()` API for use with benchmark rollout executors.
    """

    def __init__(self, policy_cfg: Any):
        self.target_temp_c = float(getattr(policy_cfg, "target_temp_c", 21.0))
        self.deadband_c = float(getattr(policy_cfg, "deadband_c", 1.0))

        self.availability_on = float(getattr(policy_cfg, "availability_on", 2.0))

        # Outlet temperature setpoints (mode-dependent)
        self.outlet_temp_heating_c = float(
            getattr(
                policy_cfg,
                "outlet_temp_heating_c",
                getattr(policy_cfg, "heating_coil_setpoint_c", 35.0),
            )
        )
        self.outlet_temp_cooling_c = float(
            getattr(
                policy_cfg,
                "outlet_temp_cooling_c",
                getattr(policy_cfg, "cooling_coil_setpoint_c", 12.0),
            )
        )

        # Fan PI config [kg/s]
        self.fan_base_kg_s = float(getattr(policy_cfg, "fan_base_kg_s", 1.0))
        self.kp = float(getattr(policy_cfg, "kp", 0.0))
        self.ki = float(getattr(policy_cfg, "ki", 0.0))
        self.integral_limit = float(getattr(policy_cfg, "integral_limit", 1e6))
        self.fan_min_cfg = getattr(policy_cfg, "fan_min_kg_s", None)
        self.fan_max_cfg = getattr(policy_cfg, "fan_max_kg_s", None)

        sched_cfg = getattr(policy_cfg, "target_schedule", None)
        enabled = bool(getattr(sched_cfg, "enabled", False)) if sched_cfg is not None else False
        weekend_days_raw = (
            getattr(sched_cfg, "weekend_days", [1, 7]) if sched_cfg is not None else [1, 7]
        )
        weekend_days = (
            {int(x) for x in weekend_days_raw}
            if isinstance(weekend_days_raw, (list, tuple))
            else {1, 7}
        )
        self.target_schedule = TargetSchedule(
            enabled=enabled,
            weekend_days=weekend_days,
            weekend_target_c=float(getattr(sched_cfg, "weekend_target_c", self.target_temp_c))
            if sched_cfg is not None
            else self.target_temp_c,
            weekday_target_c=float(getattr(sched_cfg, "weekday_target_c", self.target_temp_c))
            if sched_cfg is not None
            else self.target_temp_c,
            weekday_setback_target_c=float(
                getattr(sched_cfg, "weekday_setback_target_c", self.target_temp_c)
            )
            if sched_cfg is not None
            else self.target_temp_c,
            weekday_setback_start_hour=float(
                getattr(sched_cfg, "weekday_setback_start_hour", 9.0)
            )
            if sched_cfg is not None
            else 9.0,
            weekday_setback_end_hour=float(
                getattr(sched_cfg, "weekday_setback_end_hour", 16.0)
            )
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

        self._pi_state = PIState(integral=0.0)

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
            if (
                hasattr(env, "action_space")
                and hasattr(env.action_space, "low")
                and hasattr(env.action_space, "high")
            ):
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

    def _scheduled_target_c(self, *, obs_arr: np.ndarray) -> float:
        if not self.target_schedule.enabled:
            return float(self.target_temp_c)
        if self._tod_idx is None or self._dow_idx is None:
            return float(self.target_temp_c)
        if self._tod_idx >= len(obs_arr) or self._dow_idx >= len(obs_arr):
            return float(self.target_temp_c)

        hour = float(obs_arr[int(self._tod_idx)]) % 24.0
        day = int(round(float(obs_arr[int(self._dow_idx)])))

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

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        if self._idxs is None or self._temp_idx is None:
            raise RuntimeError("Policy is not bound to an env; call bind_env(env) first.")

        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        tz = float(obs_arr[int(self._temp_idx)]) if int(self._temp_idx) < len(obs_arr) else float("nan")
        target_now = float(self._scheduled_target_c(obs_arr=obs_arr))
        deadband = float(self.deadband_c)

        if tz < target_now - deadband:
            mode: UnitaryPIMode = "heating"
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
            mode=mode,
        )

        n_act = max(
            max(self._idxs.idx_outlet_nodes, default=-1),
            max(self._idxs.idx_fans, default=-1),
            max(self._idxs.idx_avail, default=-1),
        ) + 1
        action_cmd = np.zeros((int(n_act),), dtype=float)

        for idx in self._idxs.idx_avail:
            action_cmd[int(idx)] = float(self.availability_on)

        for idx in self._idxs.idx_fans:
            action_cmd[int(idx)] = float(
                np.clip(float(fan_cmd), float(self._fan_low), float(self._fan_high))
            )

        if mode == "heating":
            outlet_sp = float(self.outlet_temp_heating_c)
        elif mode == "cooling":
            outlet_sp = float(self.outlet_temp_cooling_c)
        else:
            outlet_sp = float(target_now)

        for idx in self._idxs.idx_outlet_nodes:
            action_cmd[int(idx)] = float(outlet_sp)

        return action_cmd, None

