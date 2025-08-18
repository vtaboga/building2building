"""Here, we define an environment from a buildingconfig just like
create_simulator.py, but we use a representation of the action and observation
space that is compatible with a less "black box" analysis of what each
observation and action is associated with.

We split the observation space and action space into interioceptive (node-local)
and exterioceptive(building-global) to make the environment compatible with
morphology agnostic RL techniques.

Here, the unit "cells" of our "robot" are the thermal zones and the thermostats.

We then associate, for instance, zone temperatures to zones and we associate
actuator state and actions to thermostats.

"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Self, Type, TypeVar, Union, reveal_type

import gymnasium
import minergym.environment
import numpy as np
import rdflib
from building2building.simulator.action_spaces import (
    DualSetpoint,
    SingleCooling,
    SingleHeating,
    SingleHeatingOrCooling,
    ThermostatSetpoint,
    get_controllable_setpoints,
)
from building2building.types import BuildingConfig, RewardType
from gymnasium.spaces import Box, Dict, Space, Tuple
from minergym.ontology import Ontology, UndirectedGraph
from minergym.simulation import (
    ActuatorHole,
    EnergyPlusSimulation,
    FunctionHole,
    MeterHole,
    VariableHole,
)
from minergym.simulation import (
    api as epapi,
)

from .transform_utils import (
    Transform,
    TransformCyclical,
    TransformDict,
    TransformIdentity,
    TransformInverse,
)

logger = logging.getLogger(__name__)


def zone_center(ont: Ontology, zone: rdflib.Node) -> np.ndarray:
    """Given a zone name, find its center point by computing the mean of all the
    centers of all its surfaces."""
    surfaces = ont.zone_surfaces(zone)

    logger.debug(f"there are {len(surfaces)} surfaces")

    center = np.zeros((3,))

    for surface in surfaces:
        points = ont.surface_vertices(surface)
        surface_center = np.zeros((3,))
        for point in points:
            surface_center += np.array(point)
            center += surface_center / len(points)

    return center / len(surfaces)


@dataclass(frozen=True)
class ZoneContext:
    center: np.ndarray


@dataclass(frozen=True)
class ActuatorContext:
    thermostat: ThermostatSetpoint


NodeContext = Union[ZoneContext, ActuatorContext]


MorphologyContext = dict[str, NodeContext]


def morphology_context(ont: Ontology) -> MorphologyContext:
    contexes = {}

    for z in ont.zones():
        center = zone_center(ont, z)

        contexes[z.toPython()] = ZoneContext(center=center)

    for sp in set(itertools.chain(*get_controllable_setpoints(ont).values())):
        contexes[sp.name] = ActuatorContext(sp)

    return contexes


@dataclass
class TransformListToArray(Transform[list, Box]):
    _domain: Any
    _codomain: Any

    def domain(self):
        return self._domain

    def codomain(self):
        return self._codomain

    def __call__(self, obj):
        return np.array(obj)

    def reverse(self, obj):
        return obj.tolist()


@dataclass
class TransformListToArrayShift(Transform[list, Box]):
    _domain: Any
    _codomain: Any

    def domain(self):
        return self._domain

    def codomain(self):
        return self._codomain

    def __call__(self, t):
        return np.array([t[0], t[1] - t[0]])

    def reverse(self, obj):
        return [float(obj[0]), float(obj[0] + obj[1])]


def thermostat_tst(t: ThermostatSetpoint) -> Transform[Any, Space]:
    match t:
        case DualSetpoint():
            return TransformListToArrayShift(
                [
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.heating_schedule
                    ),
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.cooling_schedule
                    ),
                ],
                Box(
                    np.array([15.0, 1.0]),
                    np.array([25.0, 15.0]),
                ),
            )

        case SingleHeating():
            return TransformListToArray(
                [ActuatorHole("Schedule:Compact", "Schedule Value", t.schedule)],
                Box(np.array([15]), np.array([25])),
            )

        case SingleCooling():
            return TransformListToArray(
                ActuatorHole("Schedule:Compact", "Schedule Value", t.schedule),
                Box(np.array([16]), np.array([40])),
            )

        case SingleHeatingOrCooling():
            return TransformListToArrayShift(
                [
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.heating_schedule
                    ),
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.cooling_schedule
                    ),
                ],
                Box(
                    np.array([15.0, 1.0]),
                    np.array([25.0, 15.0]),
                ),
            )

        case _:
            raise Exception(f"Should be unreachable. Got a {t}")


# To use multiprocessing to collect many trajectories in parallel, we need to be
# able to pickle an environment. For that, it must contain no references to
# local functions with closures. To solve that, we lambda lift everything.


def lifted_current_time(state):
    return epapi.exchange.current_time(state)


def lifted_day_of_year(state):
    return epapi.exchange.day_of_year(state)


def observation_tst(
    ctx: MorphologyContext,
) -> Transform:
    """Return a TST for the building's observation space."""

    def thing(k: str, v: NodeContext) -> Transform[Any, Space]:
        match v:
            case ActuatorContext():
                return thermostat_tst(v.thermostat)
            case ZoneContext():
                return TransformListToArray(
                    [VariableHole("ZONE AIR TEMPERATURE", k)],
                    Box(np.array([0.0]), np.array([100.0])),
                )

    proprio_tst = TransformDict({k: thing(k, v) for k, v in ctx.items()})

    exterio_tst = TransformDict(
        {
            "environment_temp": TransformListToArray(
                [
                    VariableHole(
                        "SITE OUTDOOR AIR DRYBULB TEMPERATURE",
                        "ENVIRONMENT",
                    )
                ],
                Box(np.array([0.0]), np.array([100.0])),
            ),
            "time": TransformDict(
                {
                    "current_time": TransformCyclical(
                        FunctionHole(lifted_current_time),
                        1,
                        25,
                    ),
                    "day_of_year": TransformCyclical(
                        FunctionHole(lifted_day_of_year),
                        1,
                        366,
                    ),
                }
            ),
            "energy": TransformDict(
                {
                    "electricity": TransformListToArray(
                        [MeterHole("Electricity:HVAC")],
                        Box(np.array([0.0]), np.array([1000.0])),
                    ),
                    "natural_gas": TransformListToArray(
                        [MeterHole("NaturalGas:HVAC")],
                        Box(np.array([0.0]), np.array([1000.0])),
                    ),
                }
            ),
        }
    )

    obs_tst = TransformDict(
        {
            "proprioceptive": proprio_tst,
            "exterioceptive": exterio_tst,
        }
    )

    return obs_tst


def action_tst(ctx: MorphologyContext) -> Transform[Any, Space]:
    def thing(c: NodeContext) -> Transform[Any, Space]:
        match c:
            case ZoneContext():
                return TransformListToArray(
                    [],
                    Box(np.array([]), np.array([])),
                )
            case ActuatorContext():
                return thermostat_tst(c.thermostat)

    return TransformDict({k: thing(v) for k, v in ctx.items()})


@dataclass
class MakeEnergyPlus:
    path_to_building: Path
    path_to_weather: Path
    observation_template: Any
    action_template: Any
    verbose: bool

    def __call__(self) -> EnergyPlusSimulation:
        return EnergyPlusSimulation(
            self.path_to_building,
            self.path_to_weather,
            self.observation_template,
            self.action_template,
            verbose=self.verbose,
        )


@dataclass
class Reward:
    def __call__(self, raw_obs) -> float:
        energy = raw_obs["exterioceptive"]["energy"]

        return -(energy["electricity"][0] + energy["natural_gas"][0])


def create_morph_env(
    config: BuildingConfig, verbose: bool = False
) -> minergym.environment.EnergyPlusEnvironment:
    ont = Ontology.from_json(config.path_to_building)
    ctx = morphology_context(ont)

    o_tst = observation_tst(ctx)
    a_tst = action_tst(ctx)

    make_energyplus = MakeEnergyPlus(
        config.path_to_building,
        config.path_to_weather,
        o_tst.domain(),
        a_tst.domain(),
        verbose,
    )

    reward = Reward()
    env = minergym.environment.EnergyPlusEnvironment(
        make_energyplus,
        reward,  # TODO: implement an actual reward function.
        o_tst.codomain(),
        o_tst,
        a_tst.codomain(),
        TransformInverse(a_tst),
    )
    env.metadata["config"] = config
    return env


# The ModuMorph paper creates the node wise morphology context by traversing the
# graph through depth first search and by encoding position and angles relative
# to a node's parent. In buildings, we can't do that so these procedures are
# essentially useless for now.

T = TypeVar("T")


@dataclass
class Tree(Generic[T]):
    value: T
    children: list[Tree[T]]


def depth_first_search(entry: T, g: UndirectedGraph[T]) -> Tree[T]:
    visited = []

    def go(e):
        visited.append(e)
        children = []
        for neighbor in g[e]:
            if neighbor not in visited:
                children.append(go(neighbor))
        return Tree(e, children)

    return go(entry)
