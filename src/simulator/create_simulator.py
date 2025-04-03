import gymnasium as gym
import typing

from src.simulator.utils import setup_energyplus_path
from src.simulator.simulation import EnergyPlusSimulation
from src.simulator.environment import EnergyPlusEnvironment
from src.simulator.rewards import base_reward_function
from src.simulator import query_info, config
from src.simulator.observation_spaces import observation_transform, create_observation_space
from src.simulator.action_spaces import action_transform, create_action_space


def create_simulator(path_to_building: str, path_to_weather: str) -> gym.Env:
    """
    Create a simulator for a given building and weather file.
    """

    obs_template = {}
    rdf = query_info.rdf_from_json(path_to_building)
    # Add observations
    config.auto_add_time(rdf, obs_template)
    config.auto_add_temperature(rdf, obs_template)
    config.auto_add_energy(rdf, obs_template)
    config.auto_add_weather(rdf, obs_template)
    # Add actuators
    actuators = config.auto_get_actuators(rdf)

    print(f"{obs_template=}")
    print(f"{actuators=}")

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
        )

    gymenv = EnergyPlusEnvironment[typing.Any, typing.Any](
        make_energyplus,
        base_reward_function,
        observation_space,
        observation_transform,
        action_space,
        action_transform,
    )

    return gymenv


