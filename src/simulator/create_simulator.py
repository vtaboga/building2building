import gymnasium as gym
import numpy as np
import typing

from src.simulator.utils import setup_energyplus_path
from src.simulator.simulation import EnergyPlusSimulation
from src.simulator.environment import EnergyPlusEnvironment
from src.simulator.rewards import reward_function
from src.simulator import query_info, config





def to_list(obs):
    """Extract from a raw observation the values we are interested in"""
    pre_obs = (
        # temperatures
        [v for zone, v in obs["temperature"].items()] +
        # temps
        [obs["time"]["current_time"]]
    )
    return pre_obs


def observation_transform(obs):
    """Extract from a raw observation the values we are interested in. Turn
    it into an array"""

    obs = np.array(to_list(obs))
    return obs


def action_transform(act):
    """The heating setpoint acts as some kind of "lower temperature bound" on
    the environment. The policy will produce, instead of a cooling setpoint,
    cooling - heating. This means the raw actions will always be consistent
    (heating < cooling).

    """

    return {
        "Space 1 ZN Thermostat Schedule": act[0],
        #"cooling_sch": act[0] + act[1],
    }

def create_simulator(path_to_building: str, path_to_weather: str, obs_names: list[str]=None, actuator_names: list[str]=None) -> gym.Env:
    """
    Create a simulator for a given building and weather file.
    """

    print(f"{path_to_building=}")
    print(f"{path_to_weather=}")


    obs_template = {}
    rdf = query_info.rdf_from_json(path_to_building)
    config.auto_add_time(rdf, obs_template)
    config.auto_add_temperature(rdf, obs_template)
    actuators = config.auto_get_actuators(rdf)

    print(f"{obs_template=}") # Look at what those are!
    print(f"{actuators=}")


    observation_mock = to_list(obs_template)
    observation_space = gym.spaces.Box(
        np.array([0.0] * len(observation_mock)),
        np.array([30.0] * len(observation_mock)),
    )

    action_space = gym.spaces.Box(
        np.array([0.0, 0.0]),
        np.array([30.0, 30.0]),
    )

    obs_template = {}
    rdf = query_info.rdf_from_json(path_to_building)
    config.auto_add_time(rdf, obs_template)
    config.auto_add_temperature(rdf, obs_template)
    actuators = config.auto_get_actuators(rdf)

    print("actuators")
    print(actuators)

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
        reward_function,
        observation_space,
        observation_transform,
        action_space,
        action_transform,
    )

    return gymenv


