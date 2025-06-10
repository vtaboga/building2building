import gymnasium as gym
import typing

from building2building.env import setup_energyplus_path
from building2building.simulator.simulation import EnergyPlusSimulation, ActuatorHole
from building2building.simulator.environment import EnergyPlusEnvironment
from building2building.simulator.rewards import base_reward_function, barrier_reward_function
from building2building.simulator import query_info, config
from building2building.simulator.observation_spaces import observation_transform, create_observation_space
from building2building.simulator.action_spaces import action_transform, create_action_space, get_controllable_setpoints_rdf
from building2building.core.run_manager import RunManager

def create_simulator(path_to_building: str, path_to_weather: str, building_characteristics: dict, reward_type: str, energy_weight: float = 1.0, run_manager: RunManager = None) -> gym.Env:
    """
    Create a simulator for a given building and weather file.
    
    Args:
        path_to_building: Path to the building file (epJSON)
        path_to_weather: Path to the weather file (epw)
        building_characteristics: Dictionary with building characteristics
        reward_type: Type of reward function to use
        run_manager: Optional RunManager to handle logging and output directories
    
    Returns:
        gym.Env: EnergyPlus environment
    """

    obs_template = {}
    rdf = query_info.rdf_from_json(path_to_building)
    # Add observations
    config.auto_add_time(rdf, obs_template)
    config.auto_add_temperature(rdf, obs_template)
    config.auto_add_energy(rdf, obs_template)
    config.auto_add_weather(rdf, obs_template)

    setpoints = get_controllable_setpoints_rdf(rdf)
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
        return EnergyPlusSimulation(
            path_to_building,
            path_to_weather,
            obs_template,
            actuators,
            run_manager=run_manager,
            verbose=False
        )


    def reward_function(obs):
        if reward_type == "barrier":
            return barrier_reward_function(obs, setpoints, building_characteristics, energy_weight)
        elif reward_type == "base":
            return base_reward_function(obs, setpoints, building_characteristics, energy_weight)
        else:
            raise ValueError(f"Invalid reward type: {reward_type}")
    

    gymenv = EnergyPlusEnvironment[typing.Any, typing.Any](
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


