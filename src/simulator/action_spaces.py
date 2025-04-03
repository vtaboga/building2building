import numpy as np
import typing
import gymnasium as gym

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

def create_action_space(actuators) -> gym.spaces.Box:
    """Create a Gymnasium action space based on the available actuators."""
    return gym.spaces.Box(
        np.array([12.0, 12.0]),
        np.array([40.0, 40.0]),
    )
