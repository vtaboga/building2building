"""The simulation module provides a .step api, but without a reward function and
without a well shaped observation and action space. This module seeks to wrap
the more "primitive" api into a gymnasium environment that we can pass without
trouble to any gymnasium consumer.

"""


import typing
import simulator.simulation as simulation
import gymnasium


ObsType = typing.TypeVar("ObsType")
ActType = typing.TypeVar("ActType")


class EnergyPlusEnvironment(gymnasium.Env, typing.Generic[ObsType, ActType]):
    """What does environment.EnergyPlusEnvironment have that
    simulation.EnergyPlusSimulation doesn't?

    1. The ability to be reset()
    2. A well defined observation and action space.
    3. A well defined reward function.

    While this class wraps all of those components together, the user will still
    have to provide their own reward function, action & observation reshaper and
    a way to initialize an EnergyPlusSimulation object.

    """

    metadata = {"render_modes": []}

    """This function will be called each time we need to .reset() the
    environment."""
    make_energyplus: typing.Callable[[], simulation.EnergyPlusSimulation]

    reward_function: typing.Callable[
        [typing.Any],  # Raw observation
        float,  # Reward value
    ]

    observation_space: gymnasium.Space[ObsType]

    """This function will be called to transform the output of the raw
    EnergyPlus controller into a point of the observation_space."""
    observation_transform: typing.Callable[
        [typing.Any],  # Raw observation
        ObsType,  # Observation space
    ]

    action_space: gymnasium.Space[ActType]
    """This function will be called to transform a point in the action_space a
    raw EnergyPlus action."""
    action_transform: typing.Callable[
        [ActType],  # Action space
        typing.Any,  # Raw action
    ]

    def __init__(
        self,
        make_energyplus: typing.Callable[[], simulation.EnergyPlusSimulation],
        reward_fn: typing.Callable[[typing.Any], float],
        observation_space: gymnasium.Space[ObsType],
        observation_transform: typing.Callable[[typing.Any], ObsType],
        action_space: gymnasium.Space[ActType],
        action_transform: typing.Callable[[ActType], typing.Any],
    ):
        super(EnergyPlusEnvironment, self).__init__()
        self.make_energyplus = make_energyplus
        self.reward_fn = reward_fn
        self.observation_space = observation_space
        self.observation_transform = observation_transform

        self.action_space = action_space
        self.action_transform = action_transform

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> typing.Tuple[typing.Any, dict[str, typing.Any]]:
        super().reset()
        self.ep = self.make_energyplus()
        obs, over = self.ep.start()
        return self.observation_transform(obs), {"raw_observation": obs}

    def step(self, action) -> typing.Tuple[ObsType, float, bool, bool, typing.Any]:
        # Do something about the actions
        a = self.action_transform(action)
        obs, finished = self.ep.step(a)
        # print(f"{obs}, {finished}")
        if not finished:
            transformed_obs = self.observation_transform(obs)
            self.last_obs = transformed_obs
            return (
                transformed_obs,
                self.reward_fn(obs),
                False,
                False,
                {"raw_observation": obs},
            )
        else:
            return (self.last_obs, 0.0, True, False, {})
