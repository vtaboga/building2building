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

from building2building.simulator.transform_utils import (
    Transform,
    TransformCompose,
    TransformDict,
    TransformDictSpace,
    TransformIdentity,
    TransformListToArray,
    TransformMonoList,
    TransformScalarToArray,
    transform_flatten,
)


# We redefine those methods as toplevel functions so that pickle is able to
# serialize references to them.
def lifted_current_time(state):
    return api.exchange.current_time(state)


def lifted_day_of_year(state):
    return api.exchange.day_of_year(state)


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


def flat_observation_info(ont: Ontology) -> ObservationInfo:
    """Create the observation transform with codomain a box with appropriate
    bounds for each variable type.

    -    The observation space contains in order:
    -    - Zone Air Temperatures (one per zone) [-50°C, 50°C]
    -    - Outdoor Air Temperature [-50°C, 50°C]
    -    - Outdoor Air Relative Humidity [0%, 100%]
    -    - Current Time of Day [0, 24]
    -    - Day of Year [1, 366]
    -    - HVAC Electricity Consumption [0, inf]
    -    - HVAC Natural Gas Consumption [0, inf]

    """

    template = {
        "temperature": {
            z.toPython(): (
                f"ZONE AIR TEMPERATURE {z.toPython()}",
                VariableHole("ZONE AIR TEMPERATURE", z.toPython()),
                (-50.0, 50.0),
            )
            for z in ont.zones()
        },
        "time": {
            "time_of_day": (
                "time_of_day",
                FunctionHole(lifted_current_time),
                (1.0, 25.0),
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
                MeterHole("NaturalGas:HVAC"),
                (0.0, float("inf")),
            ),
            "electricity": (
                "energy_electricity",
                MeterHole("Electricity:HVAC"),
                (0.0, float("inf")),
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


def dict_observation_info(ont: Ontology) -> Transform:
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
                        MeterHole("NaturalGas:HVAC"), 0.0, float("inf")
                    ),
                    "electricity": TransformScalarToArray(
                        MeterHole("Electricity:HVAC"),
                        0.0,
                        float("inf"),
                    ),
                }
            ),
        }
    )
