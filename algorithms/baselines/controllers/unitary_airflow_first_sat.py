from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


UnitaryAirflowFirstMode = Literal["heating", "cooling", "deadband"]


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

