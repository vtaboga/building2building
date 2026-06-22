"""Pins the NormalizeObservation wrapper contract.

Asserts that observations are mapped affinely into [0, 1] using the
observation-space bounds, that the transformation is invertible (denormalize
round-trips), that a zero-range dimension raises at construction, and that
``reset()`` rebuilds the bounds when the underlying space changes.  A real-env
companion test confirms the wrapper produces in-range observations on a
production env.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import building2building.api as api_mod
from building2building.simulator.wrappers import NormalizeObservation
from building2building.types import RewardConfig

_FILLED_REWARD = RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0)


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


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
def test_normalize_observation_real_env(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    """Companion: wrapper must map the real env's observation_space to [0, 1].

    Tests structural invariants without calling reset() — the fixture env's
    EnergyPlus simulation is only exercised by long tests.
    """
    _patch_registry(monkeypatch, fixture_registry)
    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=_FILLED_REWARD,
        max_episode_steps=4,
    )
    try:
        wrapped = NormalizeObservation(env)
        # observation_space must be [0, 1] in every dimension
        np.testing.assert_array_equal(wrapped.observation_space.low, 0.0)
        np.testing.assert_array_equal(wrapped.observation_space.high, 1.0)
        assert wrapped.observation_space.shape == env.unwrapped.observation_space.shape
        # denormalize must recover the original bounds
        norm_low = wrapped.observation(env.unwrapped.observation_space.low)
        norm_high = wrapped.observation(env.unwrapped.observation_space.high)
        np.testing.assert_allclose(norm_low, 0.0, atol=1e-5)
        np.testing.assert_allclose(norm_high, 1.0, atol=1e-5)
    finally:
        env.close()


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
