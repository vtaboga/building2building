from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIState:
    integral: float = 0.0


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
) -> float:
    err = float(target_c - tz_c)

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

