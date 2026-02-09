import logging
import urllib.parse
from dataclasses import dataclass
from typing import Any, Sequence

import minergym.ontology as ontology
import numpy as np
from gymnasium.spaces import Box, Dict
from minergym.simulation import ActuatorHole
from rdflib.term import Node

from b2b.types import ActuatorDescription

from .transform_utils import (
    Transform,
    TransformConcat,
    TransformDictSpace,
    TransformList,
    TransformListToArray,
    TransformListToArrayShift,
)

logger = logging.getLogger(__name__)


# Stub type for compatibility with reward functions
# TODO: Remove when rewards are refactored to not use thermostat setpoints
ThermostatSetpoint = Any


def hvac_actuators_transform(
    hvac_actuators: Sequence[ActuatorDescription],
) -> Transform[list, Box]:
    """
    Build an action transform from a list of ActuatorDescription instances.
    """

    holes: list[ActuatorHole] = []
    lows: list[float] = []
    highs: list[float] = []

    for a in hvac_actuators:
        holes.append(ActuatorHole(a.component_type, a.control_type, a.component_name))
        lows.append(a.lower_bound)
        highs.append(a.upper_bound)

    if not holes:
        raise ValueError(
            "hvac_actuators is empty; cannot build HVAC actuator action space"
        )

    return TransformListToArray(
        holes,
        Box(low=np.asarray(lows, dtype=float), high=np.asarray(highs, dtype=float)),
    )
