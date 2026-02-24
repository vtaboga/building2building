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
        logger.warning(
            "hvac_actuators is empty; the environment will not make a lot of sense"
        )

    return TransformListToArray(
        holes,
        Box(low=np.asarray(lows, dtype=float), high=np.asarray(highs, dtype=float)),
    )


# ---------------------------------------------------------------------------
# Split action space: fixed cooling setpoints 
# ---------------------------------------------------------------------------

FIXED_CLG_SP_VALUE = 40.0


def _is_fixed_actuator(a: ActuatorDescription) -> bool:
    """VAV cooling setpoints are fixed for simulation stability."""
    return "b2b vav clg setpoint" in a.component_name.lower()


@dataclass(slots=True)
class HvacActionSpace:
    """Full EnergyPlus action space + reduced space exposed to the agent.

    Cooling setpoints are removed from the agent-facing space and pinned
    to ``FIXED_CLG_SP_VALUE`` (40 °C), following the OfficeRL convention.
    """

    full_transform: TransformListToArray
    agent_transform: TransformListToArray
    agent_actuators: list[ActuatorDescription]
    fixed_indices: list[int]
    fixed_values: list[float]

    def assemble_full_action(self, agent_action: np.ndarray | Sequence[float]) -> list[float]:
        """Expand an agent action vector into the full EnergyPlus actuator vector."""
        agent_list: list[float] = (
            agent_action.tolist()
            if hasattr(agent_action, "tolist")
            else [float(x) for x in agent_action]
        )
        if len(agent_list) != len(self.agent_actuators):
            raise ValueError(
                f"Expected agent_action of length {len(self.agent_actuators)}, "
                f"got {len(agent_list)}"
            )

        fixed_set = set(self.fixed_indices)
        n_full = len(self.fixed_indices) + len(self.agent_actuators)
        full: list[float] = [0.0] * n_full

        for idx, val in zip(self.fixed_indices, self.fixed_values):
            full[int(idx)] = float(val)

        j = 0
        for i in range(n_full):
            if i not in fixed_set:
                full[i] = agent_list[j]
                j += 1

        return full


def hvac_action_space(
    hvac_actuators: Sequence[ActuatorDescription],
) -> HvacActionSpace:
    """Build a split action space where cooling setpoints are fixed at 40 °C.

    Returns an ``HvacActionSpace`` whose ``agent_transform`` exposes only
    non-fixed actuators to the agent, while ``full_transform`` covers all
    actuators for EnergyPlus.
    """
    full_transform = hvac_actuators_transform(hvac_actuators)

    agent_actuators: list[ActuatorDescription] = []
    fixed_indices: list[int] = []
    fixed_values: list[float] = []

    for i, a in enumerate(hvac_actuators):
        if _is_fixed_actuator(a):
            fixed_indices.append(i)
            fixed_values.append(FIXED_CLG_SP_VALUE)
        else:
            agent_actuators.append(a)

    agent_transform = hvac_actuators_transform(agent_actuators)

    return HvacActionSpace(
        full_transform=full_transform,
        agent_transform=agent_transform,
        agent_actuators=agent_actuators,
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
    )
