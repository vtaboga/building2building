from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from building2building.api.rl_wrappers import wrap_env_for_rl
from building2building.simulator.wrappers import NormalizeObservation


class ActionCaptureEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.array([10.0, -5.0], dtype=np.float32),
            high=np.array([30.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([15.0, 5.0], dtype=np.float32),
            high=np.array([30.0, 25.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.metadata = {
            "observation_names": ["Zone Air Temperature Z1", "Outdoor Air Temperature"]
        }
        self.received_actions: list[np.ndarray] = []

    def reset(self, **kwargs):  # type: ignore[override]
        return np.array([20.0, 0.0], dtype=np.float32), {}

    def step(self, action):  # type: ignore[override]
        self.received_actions.append(np.array(action, dtype=np.float32))
        return np.array([20.0, 0.0], dtype=np.float32), 0.0, True, False, {}


@pytest.mark.quick
def test_wrap_env_for_rl_composition_order() -> None:
    env = ActionCaptureEnv()
    wrapped = wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)

    assert isinstance(wrapped, NormalizeObservation)
    assert isinstance(wrapped.env, gym.wrappers.RescaleAction)
    np.testing.assert_allclose(wrapped.observation_space.low, np.array([0.0, 0.0]))
    np.testing.assert_allclose(wrapped.observation_space.high, np.array([1.0, 1.0]))
    np.testing.assert_allclose(wrapped.action_space.low, np.array([-1.0, -1.0]))
    np.testing.assert_allclose(wrapped.action_space.high, np.array([1.0, 1.0]))


@pytest.mark.quick
def test_wrap_env_for_rl_action_round_trip() -> None:
    env = ActionCaptureEnv()
    wrapped = wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)
    wrapped.reset()

    wrapped.step(np.array([-1.0, -1.0], dtype=np.float32))
    wrapped.step(np.array([1.0, 1.0], dtype=np.float32))
    wrapped.step(np.array([0.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(env.received_actions[0], np.array([15.0, 5.0]))
    np.testing.assert_allclose(env.received_actions[1], np.array([30.0, 25.0]))
    np.testing.assert_allclose(env.received_actions[2], np.array([22.5, 15.0]))


@pytest.mark.quick
def test_wrap_env_for_rl_flags_are_independent() -> None:
    env_action_only = ActionCaptureEnv()
    wrapped_action_only = wrap_env_for_rl(
        env_action_only, normalize_obs=False, rescale_action=True
    )
    np.testing.assert_allclose(
        wrapped_action_only.observation_space.low, env_action_only.observation_space.low
    )
    np.testing.assert_allclose(wrapped_action_only.action_space.low, np.array([-1.0, -1.0]))
    np.testing.assert_allclose(
        wrapped_action_only.action_space.high, np.array([1.0, 1.0])
    )

    env_obs_only = ActionCaptureEnv()
    wrapped_obs_only = wrap_env_for_rl(
        env_obs_only, normalize_obs=True, rescale_action=False
    )
    np.testing.assert_allclose(wrapped_obs_only.observation_space.low, np.array([0.0, 0.0]))
    np.testing.assert_allclose(wrapped_obs_only.observation_space.high, np.array([1.0, 1.0]))
    np.testing.assert_allclose(
        wrapped_obs_only.action_space.low, env_obs_only.action_space.low
    )
    np.testing.assert_allclose(
        wrapped_obs_only.action_space.high, env_obs_only.action_space.high
    )


@pytest.mark.quick
def test_wrap_env_for_rl_metadata_passthrough() -> None:
    env = ActionCaptureEnv()
    wrapped = wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)
    assert wrapped.metadata["observation_names"] == env.metadata["observation_names"]
