"""When the environment samples EnergyPlus: HVAC system or zone timesteps.

minergym hands control to the agent from
``callback_after_predictor_after_hvac_managers``, which EnergyPlus calls at
every **HVAC system timestep**. When loads change quickly, EnergyPlus splits a
zone timestep (5 minutes at 12 timesteps per hour) into shorter system
timesteps (down to 1 minute). Each of them is then one ``env.step()``, so:

* steps are not all ``60 / timesteps_per_hour`` minutes long, and an episode
  of ``task_config.expected_steps()`` steps covers less than its run period;
* ``time_of_day`` reads ``current_time()``, which includes the system time
  elapsed inside the zone timestep, and can go backwards between steps;
* the HVAC energy meters report the current zone timestep's value at every
  system timestep, so energy (and the energy term of the reward) is counted
  again on every extra step.

``sampling="hvac_timestep"`` keeps that behavior, which is the default and the
one the v1.0.0 results were produced with. ``sampling="zone_timestep"``
samples once per completed zone timestep instead:

* the observation/action handshake runs at
  ``callback_end_zone_timestep_after_zone_reporting``, so every step is one
  zone timestep and the meters report exactly that interval;
* the last action is re-applied after the predictor at every system timestep
  of the next zone timestep, so the actuators hold their value throughout;
* the clock is the physical end of the completed zone timestep,
  ``hour + zone_time_step_number * zone_time_step`` (it reads 24.0 at the
  last step of a day; ``day_of_year`` changes on the next one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import optree
from minergym.simulation import EnergyPlusSimulation, FunctionHole, api

from building2building.simulator.observation_spaces import lifted_current_time

Sampling = Literal["hvac_timestep", "zone_timestep"]
SAMPLING_MODES: tuple[str, ...] = ("hvac_timestep", "zone_timestep")


def check_sampling(sampling: str) -> None:
    """Raise ``ValueError`` unless *sampling* is one of :data:`SAMPLING_MODES`."""
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {SAMPLING_MODES}, got {sampling!r}")


@dataclass(frozen=True)
class ZoneClock:
    """Time of day, in hours, at the end of the zone timestep just completed.

    Raises ``RuntimeError`` if the building's zone timestep is not
    *step_minutes* long, since step counts (``max_episode_steps``,
    ``expected_steps()``) assume it is.
    """

    step_minutes: int

    def __call__(self, state: Any) -> float:
        exchange = api.exchange
        step_hours = float(exchange.zone_time_step(state))
        if not math.isclose(
            step_hours, self.step_minutes / 60.0, rel_tol=0.0, abs_tol=1e-7
        ):
            raise RuntimeError(
                f"EnergyPlus zone timestep is {step_hours * 60:g} minutes; "
                f"timesteps_per_hour requires {self.step_minutes} minutes"
            )
        step_number = int(exchange.zone_time_step_number(state))
        return float(exchange.hour(state)) + step_number * step_hours


def with_zone_clock(template: Any, clock: ZoneClock) -> Any:
    """Return *template* with every ``current_time()`` reader replaced by *clock*.

    Covers the ``time_of_day`` observation and the random-schedule target and
    occupancy readers (their ``hour_reader``).
    """
    replaced = 0

    def update(hole: Any) -> Any:
        nonlocal replaced
        if not isinstance(hole, FunctionHole):
            return hole
        reader: Any = hole.function
        if reader is lifted_current_time:
            replaced += 1
            return FunctionHole(clock)
        if getattr(reader, "hour_reader", None) is lifted_current_time:
            replaced += 1
            return FunctionHole(replace(reader, hour_reader=clock))
        return hole

    result = optree.tree_map(
        update, template, is_leaf=lambda node: isinstance(node, FunctionHole)
    )
    if replaced == 0:
        raise RuntimeError("observation template has no current-time reader")
    return result


class ZoneTimestepSimulation(EnergyPlusSimulation):
    """An ``EnergyPlusSimulation`` that steps once per completed zone timestep."""

    def callback_timestep(self, _: Any) -> None:
        # The base class registers this at every HVAC system timestep; the
        # handshake runs from _completed_zone_timestep instead.
        return None

    def register_callbacks(self, ep_state: Any) -> None:
        super().register_callbacks(ep_state)
        api.runtime.callback_end_zone_timestep_after_zone_reporting(
            ep_state, self._completed_zone_timestep
        )
        api.runtime.callback_after_predictor_after_hvac_managers(
            ep_state, self._reapply_last_action
        )

    def _completed_zone_timestep(self, ep_state: Any) -> None:
        EnergyPlusSimulation.callback_timestep(self, ep_state)

    def _reapply_last_action(self, ep_state: Any) -> None:
        if (
            self._last_raw_action is None
            or not hasattr(self.state, "actuator_handles")
            or api.exchange.warmup_flag(ep_state)
        ):
            return
        handles = self.state.actuator_handles  # type: ignore[union-attr]
        for accessor in optree.tree_accessors(self._last_raw_action):
            handle = accessor(handles)
            value = float(accessor(self._last_raw_action))
            api.exchange.set_actuator_value(ep_state, handle.handle, value)
            self._last_set_actuators[handle.handle] = value
