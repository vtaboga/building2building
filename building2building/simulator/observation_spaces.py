"""Observation space construction for EnergyPlus environments.

Builds flat (``Box``) or dict-based observation spaces that expose zone
air temperatures, outdoor weather, time features, energy meters, and
optional occupancy / target-temperature signals.
"""

from ctypes import c_void_p
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from minergym.ontology import Ontology
from minergym.simulation import (
    FunctionHole,
    MeterHole,
    VariableHole,
    api,
)

from building2building.simulator.schedules import (
    RandomDailyScheduleGenerator,
    distribution_for_building_type,
    month_to_season,
)
from building2building.simulator.transform_utils import (
    Transform,
    TransformDictSpace,
    TransformScalarToArray,
    transform_flatten,
)
from building2building.types import TaskConfig, ZoneTargetTemperatureConfig


# We redefine those methods as toplevel functions so that pickle is able to
# serialize references to them.
def lifted_current_time(state):
    return api.exchange.current_time(state)


def lifted_day_of_year(state):
    return api.exchange.day_of_year(state)


def lifted_day_of_week(state):
    # EnergyPlus convention: 1=Sunday, 2=Monday, ..., 7=Saturday
    return api.exchange.day_of_week(state)


def lifted_month(state):
    return api.exchange.month(state)


def lifted_year(state):
    return api.exchange.year(state)


@dataclass
class ObservationFlattener:
    """Callable that flattens a nested observation into a 1-D NumPy array.

    Attributes:
        flatten_transform: The transform that maps a nested observation
            template to a flat sequence of scalars.
    """

    flatten_transform: Transform

    def __call__(self, obs) -> np.ndarray:
        """Flatten *obs* into a 1-D ``np.ndarray``.

        Cast to ``float32`` so the observation matches the declared
        ``Box`` space dtype (Gymnasium's ``Box`` defaults to ``float32``);
        a ``float64`` array fails ``observation_space.contains`` on the
        dtype check even when every value is within bounds.
        """
        return np.array(self.flatten_transform(obs), dtype=np.float32)


@dataclass
class ObservationInfo:
    """Bundle of observation-space metadata and flattening logic.

    Attributes:
        template: Nested template consumable by minergym to read
            EnergyPlus variables at each timestep.
        slot_names: Human-readable names for each slot in the flat
            observation vector (same order as ``space``).
        flatten: Callable that converts a raw nested observation into a
            flat ``np.ndarray``.
        space: Gymnasium ``Box`` space with per-slot bounds.
    """

    template: Any
    slot_names: list[str]
    flatten: Callable[[Any], np.ndarray]
    space: Box


@dataclass
class StateZero:
    pass


@dataclass
class StateHandle:
    handle: int


@dataclass
class DivideBy:
    child: Callable[[c_void_p], Any]
    by: float

    def __call__(self, state: c_void_p) -> float:
        sub = self.child(state)
        return sub / self.by


@dataclass
class DynamicMeter:
    """Return the value of the first present meter in its candidate list, or zero.

    When no gas consuming equipment is connected to the HVAC system of a
    building, the NaturalGas:HVAC meter is unavailable (even if we explicitely
    add it to the epJSON file). In that case, it makes sense to return zero.
    """

    candidates: list[str]
    state: None | StateZero | StateHandle = None

    def __call__(self, state: c_void_p) -> float:
        if self.state is None:
            for meter_name in self.candidates:
                han = api.exchange.get_meter_handle(state, meter_name)
                if han < 0:
                    continue
                self.state = StateHandle(han)
                break
            if self.state is None:
                self.state = StateZero()

        match self.state:
            case StateHandle(han):
                return api.exchange.get_meter_value(state, han)
            case StateZero():
                return 0.0


@dataclass
class DynamicZoneVariable:
    """Return a zone variable if present, otherwise zero."""

    variable_name: str
    zone_name: str
    state: None | StateZero | StateHandle = None

    def __call__(self, state: c_void_p) -> float:
        if self.state is None:
            han = api.exchange.get_variable_handle(
                state, self.variable_name, self.zone_name
            )
            if han < 0:
                self.state = StateZero()
            else:
                self.state = StateHandle(han)

        match self.state:
            case StateHandle(han):
                return float(api.exchange.get_variable_value(state, han))
            case StateZero():
                return 0.0


@dataclass
class DynamicTargetTemperature:
    """Return the target temperature for a zone, optionally occupancy-aware.

    In ``"occupancy"`` mode, returns the occupied setpoint when
    occupancy > 0 and the unoccupied setpoint otherwise.  When
    :attr:`zone_target` uses the ``"seasonal"`` unoccupied policy, the
    unoccupied setpoint is dispatched by the current simulation month
    (DEC/JAN/FEB → winter, JUN/JUL/AUG → summer, else shoulder).  In
    any other mode, always returns the occupied setpoint.

    Attributes:
        occupancy_reader: Reader that queries zone occupancy from the
            EnergyPlus runtime state.
        mode: ``"occupancy"`` for occupancy-dependent behaviour,
            anything else for a constant setpoint.
        zone_target: Target temperature configuration for this zone.
        month_reader: Callable that returns the current simulation
            month (1–12).  Only consulted when ``zone_target`` uses the
            ``"seasonal"`` policy; may be left ``None`` when all
            callers use ``"fixed"`` (kept for test convenience).
    """

    occupancy_reader: DynamicZoneVariable
    mode: str
    zone_target: ZoneTargetTemperatureConfig
    month_reader: Callable[[c_void_p], int] | None = None

    def __call__(self, state: c_void_p) -> float:
        if self.mode == "occupancy":
            occupancy = float(self.occupancy_reader(state))
            if occupancy > 0.0:
                return self.zone_target.occupied_c
            if self.zone_target.unoccupied_policy == "seasonal":
                if self.month_reader is None:
                    raise RuntimeError(
                        "Seasonal unoccupied policy requires a month_reader; "
                        "none was configured for this DynamicTargetTemperature."
                    )
                season = month_to_season(int(self.month_reader(state)))
                return self.zone_target.unoccupied_for_season(season)
            return self.zone_target.unoccupied_c
        return self.zone_target.occupied_c


@dataclass
class _RandomScheduleCache:
    """Per-zone cache of the active :class:`DailySchedule`.

    Stateful by design: sampling is idempotent within a simulated day
    (keyed on ``(year, day_of_year)``) so that both the target and
    occupancy observations agree.
    """

    generator: RandomDailyScheduleGenerator
    last_key: tuple[int, int] | None = None
    schedule: Any = None

    def refresh(self, state: c_void_p, month_reader, year_reader, day_of_year_reader):
        year = int(year_reader(state))
        day_of_year = int(day_of_year_reader(state))
        key = (year, day_of_year)
        if self.last_key == key and self.schedule is not None:
            return self.schedule
        season = month_to_season(int(month_reader(state)))
        self.schedule = self.generator.schedule_for(
            year=year, day_of_year=day_of_year, season=season
        )
        self.last_key = key
        return self.schedule


@dataclass
class DynamicRandomScheduleTarget:
    """Return the target temperature for a zone from a random daily schedule.

    The schedule is re-sampled once per simulated day; within a day
    the occupied setpoint is returned when the current time falls in
    the ``[arrival_h, departure_h)`` window, otherwise the unoccupied
    setpoint.
    """

    cache: _RandomScheduleCache
    month_reader: Callable[[c_void_p], int]
    year_reader: Callable[[c_void_p], int]
    day_of_year_reader: Callable[[c_void_p], int]
    hour_reader: Callable[[c_void_p], float]

    def __call__(self, state: c_void_p) -> float:
        schedule = self.cache.refresh(
            state, self.month_reader, self.year_reader, self.day_of_year_reader
        )
        hour = float(self.hour_reader(state))
        if schedule.is_occupied(hour):
            return float(schedule.occupied_c)
        return float(schedule.unoccupied_c)


@dataclass
class DynamicRandomScheduleOccupancy:
    """Return a 0/1 occupancy signal derived from the random schedule.

    Shares the same per-zone :class:`_RandomScheduleCache` as the
    matching :class:`DynamicRandomScheduleTarget` so that both
    observations are always consistent within a timestep.
    """

    cache: _RandomScheduleCache
    month_reader: Callable[[c_void_p], int]
    year_reader: Callable[[c_void_p], int]
    day_of_year_reader: Callable[[c_void_p], int]
    hour_reader: Callable[[c_void_p], float]

    def __call__(self, state: c_void_p) -> float:
        schedule = self.cache.refresh(
            state, self.month_reader, self.year_reader, self.day_of_year_reader
        )
        hour = float(self.hour_reader(state))
        return 1.0 if schedule.is_occupied(hour) else 0.0


_PEAK_HVAC_POWER_W_M2 = 200.0
"""Assumed peak HVAC power density (W/m²).  The per-timestep energy bound
is derived as ``_PEAK_HVAC_POWER_W_M2 / timesteps_per_hour`` so that it
scales automatically with the simulation timestep."""

_CONTROLLED_ZONE_TEMP_BOUND = (10.0, 45.0)
"""Zone-air-temperature observation bound (°C) for HVAC-controlled zones.
These zones are held within a comfort band, so a tight bound keeps
``NormalizeObservation`` accurate for the features the agent acts on."""

_UNCONTROLLED_ZONE_TEMP_BOUND = (-40.0, 70.0)
"""Zone-air-temperature observation bound (°C) for *uncontrolled* zones
(e.g. vented attics, garages, plenums).  These free-float with the weather
and overshoot outdoor temperature via solar gain, so they need a far wider
bound than the conditioned space (outdoor temperature itself spans
``(-30, 50)``).  Applying the conditioned bound here would put real
observations outside the declared space."""


def flat_observation_info(
    ont: Ontology,
    *,
    area: float,
    controlled_zones: list[str],
    task_config: TaskConfig,
) -> ObservationInfo:
    """Create the observation transform with codomain a box with appropriate
    bounds for each variable type.

    -    The observation space contains in order:
    -    - Zone Air Temperatures (one per zone) [-50°C, 50°C]
    -    - Outdoor Air Temperature [-50°C, 50°C]
    -    - Outdoor Air Relative Humidity [0%, 100%]
    -    - Current Time of Day [0, 24]
    -    - Day of Week [1, 7]  (1=Sunday, ..., 7=Saturday)
    -    - Day of Year [1, 366]
    -    - HVAC Electricity Consumption [0, energy_bound]  Wh/m² per timestep
    -    - HVAC Natural Gas Consumption [0, energy_bound]  Wh/m² per timestep

    """

    joules_to_watthours = 3600.0
    energy_bound = _PEAK_HVAC_POWER_W_M2 / task_config.timesteps_per_hour

    occupancy_template: dict[str, Any] = {}
    target_template: dict[str, Any] = {}
    if task_config.target_temperature_mode == "occupancy":
        for zone_name in controlled_zones:
            occupancy_reader = DynamicZoneVariable(
                "Zone People Occupant Count", zone_name
            )
            zone_target = task_config.target_for_zone(zone_name)
            occupancy_template[zone_name] = (
                f"zone_occupancy {zone_name}",
                FunctionHole(occupancy_reader),
                (0.0, 20.0),
            )
            target_template[zone_name] = (
                f"target_temperature {zone_name}",
                FunctionHole(
                    DynamicTargetTemperature(
                        occupancy_reader=occupancy_reader,
                        mode=task_config.target_temperature_mode,
                        zone_target=zone_target,
                        month_reader=lifted_month,
                    )
                ),
                (10.0, 35.0),
            )
    elif task_config.target_temperature_mode == "random_schedule":
        rs_cfg = task_config.random_schedule_config
        building_type = rs_cfg.building_type if rs_cfg is not None else None
        base_seed = rs_cfg.seed if rs_cfg is not None else 0
        distribution = distribution_for_building_type(building_type)
        for zone_idx, zone_name in enumerate(controlled_zones):
            # Decorrelate schedules across zones by offsetting the seed.
            generator = RandomDailyScheduleGenerator(
                distribution=distribution,
                base_seed=int(base_seed) + zone_idx * 997,
            )
            cache = _RandomScheduleCache(generator=generator)
            occupancy_template[zone_name] = (
                f"zone_occupancy {zone_name}",
                FunctionHole(
                    DynamicRandomScheduleOccupancy(
                        cache=cache,
                        month_reader=lifted_month,
                        year_reader=lifted_year,
                        day_of_year_reader=lifted_day_of_year,
                        hour_reader=lifted_current_time,
                    )
                ),
                (0.0, 1.0),
            )
            target_template[zone_name] = (
                f"target_temperature {zone_name}",
                FunctionHole(
                    DynamicRandomScheduleTarget(
                        cache=cache,
                        month_reader=lifted_month,
                        year_reader=lifted_year,
                        day_of_year_reader=lifted_day_of_year,
                        hour_reader=lifted_current_time,
                    )
                ),
                (10.0, 35.0),
            )

    controlled_zone_set = set(controlled_zones)
    template = {
        "temperature": {
            zone_name: (
                f"ZONE AIR TEMPERATURE {zone_name}",
                VariableHole("ZONE AIR TEMPERATURE", zone_name),
                _CONTROLLED_ZONE_TEMP_BOUND
                if zone_name in controlled_zone_set
                else _UNCONTROLLED_ZONE_TEMP_BOUND,
            )
            for zone_name in sorted(z.toPython() for z in ont.zones())
        },
        "zone_occupancy": occupancy_template,
        "target_temperature": target_template,
        "time": {
            "time_of_day": (
                "time_of_day",
                FunctionHole(lifted_current_time),
                # EnergyPlus current_time() returns fractional hours in
                # (0, 24]; the first step of each day is ~0.08, so the low
                # bound must be 0.0, not 1.0.
                (0.0, 25.0),
            ),
            "day_of_week": (
                "day_of_week",
                FunctionHole(lifted_day_of_week),
                (1.0, 7.0),
            ),
            "day_of_year": (
                "day_of_year",
                FunctionHole(lifted_day_of_year),
                (1.0, 366.0),
            ),
        },
        "outdoor": {
            "temperature": (
                "outdoor_temperature",
                VariableHole(
                    "SITE OUTDOOR AIR DRYBULB TEMPERATURE",
                    "ENVIRONMENT",
                ),
                (-30.0, 50.0),
            ),
            "humidity": (
                "outdoor_humidity",
                VariableHole("Site Outdoor Air Relative Humidity", "Environment"),
                (0.0, 100.0),
            ),
        },
        "energy": {
            "natural_gas": (
                "energy_gas",
                FunctionHole(
                    DivideBy(
                        DynamicMeter(["NaturalGas:HVAC"]), area * joules_to_watthours
                    )
                ),
                (0.0, energy_bound),
            ),
            "electricity": (
                "energy_electricity",
                FunctionHole(
                    DivideBy(
                        DynamicMeter(["Electricity:HVAC"]), area * joules_to_watthours
                    )
                ),
                (0.0, energy_bound),
            ),
        },
    }

    flattened = transform_flatten(template)

    names_tuple, flat_template, low_high = list(zip(*flattened.codomain()))
    low, high = list(zip(*low_high))

    b = Box(np.array(low), np.array(high))

    return ObservationInfo(
        template=flattened.reverse(list(flat_template)),
        slot_names=list(names_tuple),
        flatten=ObservationFlattener(
            flattened,
        ),
        space=b,
    )


def dict_observation_info(
    ont: Ontology, *, area: float, controlled_zones: list[str]
) -> Transform:
    """Build a dict-based observation transform (non-flattened).

    Unlike :func:`flat_observation_info`, this returns a nested
    ``Dict`` space suitable for agents that consume structured
    observations.

    Args:
        ont: Building ontology providing zone metadata.
        area: Building conditioned floor area (m²), used to normalise
            energy readings.
        controlled_zones: Names of the HVAC-controlled zones. Controlled
            zones get the tight :data:`_CONTROLLED_ZONE_TEMP_BOUND`;
            free-floating zones get the wider
            :data:`_UNCONTROLLED_ZONE_TEMP_BOUND`, so that real readings
            stay inside the declared space (mirrors
            :func:`flat_observation_info`).

    Returns:
        A ``Transform`` whose codomain is a Gymnasium ``Dict`` space.
    """
    controlled_zone_set = set(controlled_zones)
    return TransformDictSpace(
        {
            "temperature": TransformDictSpace(
                {
                    zone_name: TransformScalarToArray(
                        VariableHole("ZONE AIR TEMPERATURE", zone_name),
                        *(
                            _CONTROLLED_ZONE_TEMP_BOUND
                            if zone_name in controlled_zone_set
                            else _UNCONTROLLED_ZONE_TEMP_BOUND
                        ),
                    )
                    for zone_name in sorted(z.toPython() for z in ont.zones())
                }
            ),
            "time": TransformDictSpace(
                {
                    "time_of_day": TransformScalarToArray(
                        FunctionHole(lifted_current_time), 1.0, 25.0
                    ),
                    "day_of_week": TransformScalarToArray(
                        FunctionHole(lifted_day_of_week), 1.0, 7.0
                    ),
                    "day_of_year": TransformScalarToArray(
                        FunctionHole(lifted_day_of_year), 1.0, 366.0
                    ),
                }
            ),
            "outdoor": TransformDictSpace(
                {
                    "temperature": TransformScalarToArray(
                        VariableHole(
                            "SITE OUTDOOR AIR DRYBULB TEMPERATURE",
                            "ENVIRONMENT",
                        ),
                        -30.0,
                        50.0,
                    ),
                    "humidity": TransformScalarToArray(
                        VariableHole(
                            "Site Outdoor Air Relative Humidity", "Environment"
                        ),
                        0.0,
                        100.0,
                    ),
                }
            ),
            "energy": TransformDictSpace(
                {
                    "natural_gas": TransformScalarToArray(
                        FunctionHole(
                            DivideBy(DynamicMeter(["NaturalGas:HVAC"]), area * 3600.0)
                        ),
                        0.0,
                        50.0,
                    ),
                    "electricity": TransformScalarToArray(
                        FunctionHole(
                            DivideBy(DynamicMeter(["Electricity:HVAC"]), area * 3600.0)
                        ),
                        0.0,
                        50.0,
                    ),
                }
            ),
        }
    )
