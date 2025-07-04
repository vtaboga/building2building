import gymnasium as gym

from pathlib import Path

from typing import Any

from building2building.env import setup_energyplus_path
from minergym.simulation import EnergyPlusSimulation, ActuatorHole
from minergym.environment import EnergyPlusEnvironment
from building2building.simulator.rewards import base_reward_function, barrier_reward_function
import minergym.config as config
from building2building.simulator.observation_spaces import observation_transform, create_observation_space
from building2building.simulator.action_spaces import action_transform, create_action_space, get_controllable_setpoints
from minergym.ontology import Ontology
import minergym.simulation as simulation
import logging
import numpy as np


logger = logging.getLogger(__name__)


def auto_add_energy(
       ont: Ontology, obs_template: dict[str, Any]
   ) -> None:
    """Add HVAC energy consumption meters to the observation template."""
    if "energy" not in obs_template:
        obs_template["energy"] = {}

    energy = obs_template["energy"]
    # Add whole building HVAC energy meters only
    energy["HVAC_electricity"] = simulation.MeterHole("Electricity:HVAC")
    energy["HVAC_natural_gas"] = simulation.MeterHole("NaturalGas:HVAC")

def create_simulator(
        path_to_building: Path,
        path_to_weather: Path,
        building_characteristics: dict,
        reward_type: str,
        energy_weight: float = 1.0,
        eplus_output_dir: Path | None = None,
) -> gym.Env:
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

    if eplus_output_dir is None:
        eplus_output_dir = Path("eplus_output")

    obs_template = {}
    ont = Ontology.from_json(path_to_building)
    # Add observations
    config.auto_add_time(ont, obs_template)
    config.auto_add_temperature(ont, obs_template)
    auto_add_energy(ont, obs_template)
    config.auto_add_weather(ont, obs_template)

    setpoints = get_controllable_setpoints(ont)

    actuators = {}
    controlled_zones = list(setpoints.keys())
    all_zones = building_characteristics.get("zone_lists", [])
    uncontrolled_zones = [zone for zone in all_zones if zone not in controlled_zones]

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
        if eplus_output_dir is None:
            log_dir = Path("eplus_output")
        else:
            log_dir = eplus_output_dir

        sim = EnergyPlusSimulation(
            path_to_building,
            path_to_weather,
            obs_template,
            actuators,
            verbose=False,
            log_dir=log_dir,
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

    gymenv = EnergyPlusEnvironment[np.ndarray, np.ndarray](
        make_energyplus,
        reward_function,
        observation_space,
        lambda obs: observation_transform(obs, building_characteristics["area"]),
        action_space,
        lambda act: action_transform(act, actuators),
    )

    gymenv.metadata = {
        "controlled_zones": controlled_zones,
        "uncontrolled_zones": uncontrolled_zones,
        "observation_names": observation_names,
    }

    return gymenv


