import gymnasium as gym
import typing

from src.simulator.utils import setup_energyplus_path
from src.simulator.simulation import EnergyPlusSimulation, ActuatorHole
from src.simulator.environment import EnergyPlusEnvironment
from src.simulator.rewards import base_reward_function
from src.simulator import query_info, config
from src.simulator.observation_spaces import observation_transform, create_observation_space
from src.simulator.action_spaces import action_transform, create_action_space, get_controllable_setpoints_rdf
from src.core.run_manager import RunManager

def create_simulator(path_to_building: str, path_to_weather: str, building_characteristics: dict, run_manager: RunManager = None) -> gym.Env:
    """
    Create a simulator for a given building and weather file.
    
    Args:
        path_to_building: Path to the building file (epJSON)
        path_to_weather: Path to the weather file (epw)
        building_characteristics: Dictionary with building characteristics
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
    # Get controllable setpoints instead of all actuators
    setpoints = get_controllable_setpoints_rdf(rdf)
    actuators = {}
    for zone_setpoints in setpoints.values():
        for setpoint in zone_setpoints:
            actuators[setpoint['schedule_name']] = ActuatorHole(
                "Schedule:Compact",
                "Schedule Value",
                setpoint['schedule_name']
            )

    observation_space = create_observation_space(obs_template)
    action_space = create_action_space(actuators)


    def make_energyplus() -> EnergyPlusSimulation:
        # For a simulation to run, we need a building file,
        # a weather file, an observation template and the dict
        # of actuators we want to control.
        return EnergyPlusSimulation(
            path_to_building,
            path_to_weather,
            obs_template,
            actuators,
            run_manager=run_manager  # Pass the run manager to the simulation
        )


    def reward_function(obs):
        return base_reward_function(obs, setpoints, building_characteristics)

    gymenv = EnergyPlusEnvironment[typing.Any, typing.Any](
        make_energyplus,
        reward_function, 
        observation_space,
        observation_transform,
        action_space,
        lambda act: action_transform(act, actuators),
    )

    return gymenv


