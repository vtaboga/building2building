import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import minergym.config as config
import minergym.simulation as simulation
import numpy as np
from building2building.simulator.action_spaces import (
    get_controllable_setpoints,
    many_thermostats_transform,
)
from building2building.simulator.observation_spaces import flat_observation_info
from building2building.simulator.rewards import (
    BarrierReward,
    BaseReward,
    barrier_reward_function,
    base_reward_function,
)
from building2building.types import BuildingConfig
from minergym.environment import EnergyPlusEnvironment
from minergym.ontology import Ontology
from minergym.simulation import ActuatorHole, EnergyPlusSimulation

from .transform_utils import TransformInverse

logger = logging.getLogger(__name__)


@dataclass
class MakeEnergyPlus:
    """This could simply be a closure, but it wouldn't be serializable with
    pickle."""

    path_to_building: Path
    path_to_weather: Path
    observation_template: Any
    action_template: Any
    verbose: bool
    log_dir: Path

    def __call__(self) -> EnergyPlusSimulation:
        sim = EnergyPlusSimulation(
            self.path_to_building,
            self.path_to_weather,
            self.observation_template,
            self.action_template,
            verbose=self.verbose,
            log_dir=self.log_dir,
        )

        return sim


def create_simulator(building_config: BuildingConfig) -> gym.Env:
    """
    Create a simulator for a given building and weather file.

    Args:
        path_to_building: Path to the building file (epJSON)
        path_to_weather: Path to the weather file (epw)
        building_characteristics: Dictionary with building characteristics
        reward_type: Type of reward function to use
        energy_weight: Energy weight for reward calculation
        eplus_output_dir: Optional directory for EnergyPlus outputs

    Returns:
        gym.Env: EnergyPlus environment
    """

    if not isinstance(building_config, BuildingConfig):
        # If the type constraints are satisfied, it should be unreachable, but
        # this function is called through gymnasium.make, which doesn't
        # propagate type constraints.
        raise Exception(f"{building_config} should have type BuildingConfig")

    eplus_output_dir = building_config.eplus_output_dir

    ont = Ontology.from_json(building_config.path_to_building)

    # We compute the observation side stuff
    obs_info = flat_observation_info(ont)

    # Then the action side stuff
    setpoints = get_controllable_setpoints(ont)

    thermostat_list = [elem for list in setpoints.values() for elem in list]

    action_transform = many_thermostats_transform(thermostat_list)

    make_energyplus = MakeEnergyPlus(
        building_config.path_to_building,
        building_config.path_to_weather,
        obs_info.template,
        action_transform.domain(),
        verbose=False,
        log_dir=eplus_output_dir,
    )

    if building_config.reward_type == "barrier":
        reward_function = BarrierReward(
            building_config.characteristics, setpoints, building_config.energy_weight
        )
    elif building_config.reward_type == "base":
        reward_function = BaseReward(
            building_config.characteristics, setpoints, building_config.energy_weight
        )
    else:
        raise ValueError(f"Invalid reward type: {building_config.reward_type}")

    # Finally, we compute the data necessary to fillin the metadata
    controlled_zones = list(setpoints.keys())
    all_zones = building_config.characteristics.zone_lists
    uncontrolled_zones = [zone for zone in all_zones if zone not in controlled_zones]

    gymenv = EnergyPlusEnvironment[np.ndarray, np.ndarray](
        make_energyplus,
        reward_function,
        obs_info.space,
        obs_info.flatten,
        action_transform.codomain(),
        TransformInverse(action_transform),
    )

    gymenv.metadata = {
        "controlled_zones": controlled_zones,
        "uncontrolled_zones": uncontrolled_zones,
        "observation_names": obs_info.slot_names,
    }

    return gymenv
