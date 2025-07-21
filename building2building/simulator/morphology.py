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

from dataclasses import dataclass
from typing import Any, Callable, Generic, Self, Type, TypeVar, reveal_type

import gymnasium
import minergym.environment
import numpy as np
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
from minergym.ontology import Ontology
from minergym.simulation import ActuatorHole, EnergyPlusSimulation, VariableHole

A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class Transform(Generic[A, B]):
    """A tuple containing a template, a gymnasium space codifying that template
    and an isomorphism between the two represetations.

            transform
    domain <---------> codomain

    """

    domain: A
    codomain: B
    transform: Callable[[Any], Any]
    detransform: Callable[[Any], Any]

    def inverse(self):
        return Transform(self.codomain, self.domain, self.detransform, self.transform)


def transform_dict(d: dict[str, Transform[Any, Space]]) -> Transform[Any, Space]:
    template: Any = {k: v.domain for k, v in d.items()}
    space: Space = Dict({k: v.codomain for k, v in d.items()})

    def transform(a):
        return {k: v.transform(a[k]) for k, v in d.items()}

    def detransform(b):
        return {k: v.detransform(b[k]) for k, v in d.items()}

    return Transform(template, space, transform, detransform)


def transform_list(l: list[Transform[Any, Space]]) -> Transform[Any, Space]:
    def transform(a):
        return tuple(s.transform(e) for s, e in zip(l, a))

    def detransform(b):
        return [s.detransform(e) for s, e in zip(l, b)]

    return Transform(
        [v.domain for v in l],
        Tuple([v.codomain for v in l]),
        transform,
        detransform,
    )


def reward_function(thing) -> float:
    return 0.0


def action_tst(ont: Ontology) -> Transform[Any, Space]:
    thermostats = get_controllable_setpoints(ont)
    thermostats_set: set[ThermostatSetpoint] = set(
        elem for list in thermostats.values() for elem in list
    )

    thermostat_dict = {
        f"thermostat_{i}": thermostat_tst(v) for i, v in enumerate(thermostats_set)
    }

    return transform_dict(thermostat_dict)


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
    ont: Ontology,
) -> Transform:
    """Return a template,space,transform tuple for the building corresponding to
    the given ontology."""

    zones: list[str] = [z.toPython() for z in ont.zones()]

    thermostats = get_controllable_setpoints(ont)
    thermostats_set: set[ThermostatSetpoint] = set(
        elem for list in thermostats.values() for elem in list
    )

    thermostat_dict = {
        f"thermostat_{i}": thermostat_tst(v) for i, v in enumerate(thermostats_set)
    }

    temp_dict = {
        z: Transform[Any, Space](
            VariableHole("ZONE AIR TEMPERATURE", z),
            Box(0.0, 100.0),
            lambda x: x,
            lambda x: x,
        )
        for z in zones
    }

    proprio_tst = transform_dict({**temp_dict, **thermostat_dict})

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


def create_morph_env(config: BuildingConfig) -> gymnasium.Env:
    ont = Ontology.from_json(config.path_to_building)

    o_tst = observation_tst(ont)

    a_tst = action_tst(ont)

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
        reward_function,
        o_tst.codomain,
        o_tst.transform,
        a_tst.codomain,
        a_tst.detransform,
    )
