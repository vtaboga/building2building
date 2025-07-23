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
from building2building.types import BuildingConfig
from gymnasium.spaces import Box, Dict, Space, Tuple
from minergym.ontology import Ontology, UndirectedGraph
from minergym.simulation import ActuatorHole, EnergyPlusSimulation, VariableHole

from .transform_utils import Transform, transform_dict

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


def thermostat_tst(t: ThermostatSetpoint) -> Transform[Any, Space]:
    match t:
        case DualSetpoint():

            def transform(t: list[float]):
                return np.array([t[0], t[1] - t[0]])

            def detransform(a: np.ndarray):
                return [a[0], a[0] + a[1]]

            return Transform(
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
                transform,
                detransform,
            )

        case SingleHeating():
            return Transform(
                ActuatorHole("Schedule:Compact", "Schedule Value", t.schedule),
                Box(15, 25),
                lambda x: x,
                lambda x: x,
            )

        case SingleCooling():
            return Transform(
                ActuatorHole("Schedule:Compact", "Schedule Value", t.schedule),
                Box(16, 40),
                lambda x: x,
                lambda x: x,
            )

        case SingleHeatingOrCooling():

            def transform(t: list[float]):
                return np.array([t[0], t[1] - t[0]])

            def detransform(a: np.ndarray):
                return [a[0], a[0] + a[1]]

            return Transform(
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
                transform,
                detransform,
            )

        case _:
            raise Exception(f"Should be unreachable. Got a {t}")


def observation_tst(
    ctx: MorphologyContext,
) -> Transform:
    """Return a TST for the building's observation space."""

    def thing(k: str, v: NodeContext) -> Transform[Any, Space]:
        match v:
            case ActuatorContext():
                return thermostat_tst(v.thermostat)
            case ZoneContext():
                return Transform(
                    VariableHole("ZONE AIR TEMPERATURE", k),
                    Box(0.0, 100.0),
                    lambda x: x,
                    lambda x: x,
                )

    proprio_tst = transform_dict({k: thing(k, v) for k, v in ctx.items()})

    exterio_tst = transform_dict(
        {
            "environment_temp": Transform(
                VariableHole(
                    "SITE OUTDOOR AIR DRYBULB TEMPERATURE",
                    "ENVIRONMENT",
                ),
                Box(0.0, 100.0),
                lambda x: x,
                lambda x: x,
            )
        }
    )

    obs_tst = transform_dict(
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
                return Transform(
                    tuple(),
                    Tuple([]),
                    lambda x: x,
                    lambda x: x,
                )
            case ActuatorContext():
                return thermostat_tst(c.thermostat)

    return transform_dict({k: thing(v) for k, v in ctx.items()})


def create_morph_env(config: BuildingConfig) -> gymnasium.Env:
    ont = Ontology.from_json(config.path_to_building)
    ctx = morphology_context(ont)

    o_tst = observation_tst(ctx)
    a_tst = action_tst(ctx)

    def make_energyplus():
        sim = EnergyPlusSimulation(
            config.path_to_building,
            config.path_to_weather,
            o_tst.domain,
            a_tst.domain,
        )

        return sim

    return minergym.environment.EnergyPlusEnvironment(
        make_energyplus,
        lambda _: 0.0,  # TODO: implement an actual reward function.
        o_tst.codomain,
        o_tst.transform,
        a_tst.codomain,
        a_tst.detransform,
    )
