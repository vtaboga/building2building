from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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

