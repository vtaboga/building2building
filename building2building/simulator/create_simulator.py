import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import minergym.config as config
import minergym.simulation as simulation
import numpy as np
from building2building.simulator.action_spaces import (
    get_controllable_setpoints,
    template_space_transform,
)
from building2building.simulator.observation_spaces import (
    create_observation_space,
    observation_transform,
)
from building2building.simulator.rewards import (
    barrier_reward_function,
    base_reward_function,
)
from building2building.types import BuildingConfig
from minergym.environment import EnergyPlusEnvironment
from minergym.ontology import Ontology
from minergym.simulation import ActuatorHole, EnergyPlusSimulation

logger = logging.getLogger(__name__)


def auto_add_energy(ont: Ontology, obs_template: dict[str, Any]) -> None:
    """Add HVAC energy consumption meters to the observation template."""

    energy = obs_template.setdefault("energy", {})
    # Add whole building HVAC energy meters only
    energy["HVAC_electricity"] = simulation.MeterHole("Electricity:HVAC")
    energy["HVAC_natural_gas"] = simulation.MeterHole("NaturalGas:HVAC")


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
        raise Exception(f"{building_config} should have type BuildingConfig")

    eplus_output_dir = building_config.eplus_output_dir

    obs_template = {}
    ont = Ontology.from_json(building_config.path_to_building)
    # Add observations
    config.auto_add_time(ont, obs_template)
    config.auto_add_temperature(ont, obs_template)
    auto_add_energy(ont, obs_template)
    config.auto_add_weather(ont, obs_template)

    setpoints = get_controllable_setpoints(ont)

    actuators = {}
    controlled_zones = list(setpoints.keys())
    all_zones = building_config.characteristics.zone_lists
    uncontrolled_zones = [zone for zone in all_zones if zone not in controlled_zones]

    thermostat_list = [elem for list in setpoints.values() for elem in list]

    action_template, action_space, action_transform = template_space_transform(
        thermostat_list
    )

    observation_space, observation_names = create_observation_space(obs_template)

    def make_energyplus() -> EnergyPlusSimulation:
        sim = EnergyPlusSimulation(
            building_config.path_to_building,
            building_config.path_to_weather,
            obs_template,
            action_template,
            verbose=False,
        )
        # Set the log directory if provided
        if eplus_output_dir:
            sim.log_dir = eplus_output_dir
        return sim

    def reward_function(obs):
        if building_config.reward_type == "barrier":
            return barrier_reward_function(
                obs,
                building_config.characteristics,
                setpoints,
                building_config.energy_weight,
            )
        elif building_config.reward_type == "base":
            return base_reward_function(
                obs,
                building_config.characteristics,
                setpoints,
                building_config.energy_weight,
            )
        else:
            raise ValueError(f"Invalid reward type: {building_config.reward_type}")

    gymenv = EnergyPlusEnvironment[np.ndarray, np.ndarray](
        make_energyplus,
        reward_function,
        observation_space,
        lambda obs: observation_transform(obs, building_config.characteristics.area),
        action_space,
        action_transform,
    )

    gymenv.metadata = {
        "controlled_zones": controlled_zones,
        "uncontrolled_zones": uncontrolled_zones,
        "observation_names": observation_names,
    }

    return gymenv
