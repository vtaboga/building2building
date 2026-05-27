from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from building2building.simulator.wrappers import NormalizeObservation


class MutableObservationEnv(gym.Env):
    def __init__(self, low: np.ndarray, high: np.ndarray, reset_obs: np.ndarray):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_obs = reset_obs.astype(np.float32)

    def reset(self, **kwargs):  # type: ignore[override]
        return self._reset_obs.copy(), {}

    def step(self, action):  # type: ignore[override]
        return self._reset_obs.copy(), 0.0, True, False, {}


@pytest.mark.quick
def test_normalize_observation_affine_midrange() -> None:
    env = MutableObservationEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        reset_obs=np.array([20.0, 0.0], dtype=np.float32),
    )
    wrapped = NormalizeObservation(env)
    normalized = wrapped.observation(np.array([20.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(normalized, np.array([0.5, 0.5], dtype=np.float32))


@pytest.mark.quick
def test_normalize_observation_round_trip() -> None:
    env = MutableObservationEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        reset_obs=np.array([20.0, 0.0], dtype=np.float32),
    )
    wrapped = NormalizeObservation(env)
    rng = np.random.default_rng(7)
    for _ in range(25):
        sample = rng.uniform(low=env.observation_space.low, high=env.observation_space.high)
        sample = sample.astype(np.float32)
        round_trip = wrapped.denormalize(wrapped.observation(sample))
        np.testing.assert_allclose(round_trip, sample, rtol=1e-6, atol=1e-6)


@pytest.mark.quick
def test_normalize_observation_raises_on_zero_range() -> None:
    env = MutableObservationEnv(
        low=np.array([10.0, 0.0], dtype=np.float32),
        high=np.array([30.0, 0.0], dtype=np.float32),
        reset_obs=np.array([20.0, 0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="range is zero"):
        NormalizeObservation(env)


@pytest.mark.quick
def test_normalize_observation_reset_rebuilds_bounds() -> None:
    env = MutableObservationEnv(
        low=np.array([0.0], dtype=np.float32),
        high=np.array([10.0], dtype=np.float32),
        reset_obs=np.array([5.0], dtype=np.float32),
    )
    wrapped = NormalizeObservation(env)
    obs1, _ = wrapped.reset()
    np.testing.assert_allclose(obs1, np.array([0.5], dtype=np.float32))

    env.observation_space = gym.spaces.Box(
        low=np.array([10.0], dtype=np.float32),
        high=np.array([30.0], dtype=np.float32),
        dtype=np.float32,
    )
    env._reset_obs = np.array([20.0], dtype=np.float32)

    obs2, _ = wrapped.reset()
    np.testing.assert_allclose(obs2, np.array([0.5], dtype=np.float32))
    np.testing.assert_allclose(
        wrapped.observation(np.array([20.0], dtype=np.float32)),
        np.array([0.5], dtype=np.float32),
    )
