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

from b2b.simulator.transform_utils import (
    Transform,
    TransformDictSpace,
    TransformScalarToArray,
    transform_flatten,
)
from b2b.types import TaskConfig, ZoneTargetTemperatureConfig


# We redefine those methods as toplevel functions so that pickle is able to
# serialize references to them.
def lifted_current_time(state):
    return api.exchange.current_time(state)


def lifted_day_of_year(state):
    return api.exchange.day_of_year(state)


def lifted_day_of_week(state):
    # EnergyPlus convention: 1=Sunday, 2=Monday, ..., 7=Saturday
    return api.exchange.day_of_week(state)


@dataclass
class ObservationFlattener:
    flatten_transform: Transform

    def __call__(self, obs) -> np.ndarray:
        return np.array(self.flatten_transform(obs))


@dataclass
class ObservationInfo:
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
    """Return the value of the first present meter in its candidate list, or
    zero.

    When no gas consuming equipment is connected to the HVAC system of a
    building, the NaturalGas:HVAC meter is unavailable (even if we explicitely
    add it to the epJSON file). In that case, it makes sense to return zero. In the future,

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
    occupancy_reader: DynamicZoneVariable
    mode: str
    zone_target: ZoneTargetTemperatureConfig

    def __call__(self, state: c_void_p) -> float:
        if self.mode == "occupancy":
            occupancy = float(self.occupancy_reader(state))
            if occupancy > 0.0:
                return self.zone_target.occupied_c
            return self.zone_target.unoccupied_c
        return self.zone_target.occupied_c


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
    -    - HVAC Electricity Consumption [0, 50]  Wh/m² per timestep
    -    - HVAC Natural Gas Consumption [0, 50]  Wh/m² per timestep

    """

    occupancy_template = {}
    target_template = {}
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
                )
            ),
            (10.0, 35.0),
        )

    template = {
        "temperature": {
            z.toPython(): (
                f"ZONE AIR TEMPERATURE {z.toPython()}",
                VariableHole("ZONE AIR TEMPERATURE", z.toPython()),
                (-50.0, 50.0),
            )
            for z in ont.zones()
        },
        "zone_occupancy": occupancy_template,
        "target_temperature": target_template,
        "time": {
            "time_of_day": (
                "time_of_day",
                FunctionHole(lifted_current_time),
                (1.0, 25.0),
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
                (-50.0, 50.0),
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
                    DivideBy(DynamicMeter(["NaturalGas:HVAC"]), area * 3600.0)
                ),
                (0.0, 50.0),
            ),
            "electricity": (
                "energy_electricity",
                FunctionHole(
                    DivideBy(DynamicMeter(["Electricity:HVAC"]), area * 3600.0)
                ),
                (0.0, 50.0),
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


def dict_observation_info(ont: Ontology, *, area: float) -> Transform:
    return TransformDictSpace(
        {
            "temperature": TransformDictSpace(
                {
                    z.toPython(): TransformScalarToArray(
                        VariableHole("ZONE AIR TEMPERATURE", z.toPython()),
                        -50.0,
                        50.0,
                    )
                    for z in ont.zones()
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
                        -50.0,
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
                            DivideBy(
                                DynamicMeter(["NaturalGas:HVAC"]), area * 3600.0
                            )
                        ),
                        0.0,
                        50.0,
                    ),
                    "electricity": TransformScalarToArray(
                        FunctionHole(
                            DivideBy(
                                DynamicMeter(["Electricity:HVAC"]), area * 3600.0
                            )
                        ),
                        0.0,
                        50.0,
                    ),
                }
            ),
        }
    )
