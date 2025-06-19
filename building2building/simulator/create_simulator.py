import gymnasium as gym

from pathlib import Path

from typing import Any

from building2building.env import setup_energyplus_path
from building2building.simulator.simulation import EnergyPlusSimulation, ActuatorHole
from building2building.simulator.environment import EnergyPlusEnvironment
from building2building.simulator.rewards import base_reward_function, barrier_reward_function
from building2building.simulator import config
from building2building.simulator.observation_spaces import observation_transform, create_observation_space
from building2building.simulator.action_spaces import action_transform, create_action_space, get_controllable_setpoints
from building2building.core.run_manager import RunManager
from building2building.ontology import Ontology

import logging

logger = logging.getLogger(__name__)

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

    obs_template = {}
    ont = Ontology.from_json(path_to_building)
    # Add observations
    config.auto_add_time(ont, obs_template)
    config.auto_add_temperature(ont, obs_template)
    config.auto_add_energy(ont, obs_template)
    config.auto_add_weather(ont, obs_template)

    setpoints = get_controllable_setpoints(ont)
    # breakpoint()

    actuators = {}
    controlled_zones = list(setpoints.keys())
    for zone_setpoints in setpoints.values():
        for setpoint in zone_setpoints:
            actuators[setpoint['schedule_name']] = ActuatorHole(
                "Schedule:Compact",
                "Schedule Value",
                setpoint['schedule_name']
            )

    observation_space, observation_names = create_observation_space(obs_template)
    action_space = create_action_space(actuators)

    def make_energyplus() -> EnergyPlusSimulation:
        sim = EnergyPlusSimulation(
            str(path_to_building),
            str(path_to_weather),
            obs_template,
            actuators,
            verbose=False
        )

        logger.debug(f"created simulator {sim}")
        return sim


    def reward_function(obs):
        if reward_type == "barrier":
            return barrier_reward_function(obs, setpoints, building_characteristics, energy_weight)
        elif reward_type == "base":
            return base_reward_function(obs, setpoints, building_characteristics, energy_weight)
        else:
            raise ValueError(f"Invalid reward type: {reward_type}")
    

    gymenv = EnergyPlusEnvironment[Any, Any](
        make_energyplus,
        reward_function, 
        observation_space,
        lambda obs: observation_transform(obs, building_characteristics["area"]),
        action_space,
        lambda act: action_transform(act, actuators),
        building_characteristics,
        controlled_zones,
        observation_names
    )

    return gymenv


