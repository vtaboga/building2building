"""Pins the ``wrap_env_for_rl`` composition and rescaling contract.

Asserts that ``normalize_obs`` and ``rescale_action`` flags are independent,
that their composition order places ``NormalizeObservation`` outermost, that
actions in [-1, 1] are correctly mapped to physical actuator ranges, and that
``env.metadata`` is preserved through the wrapper stack.  A real-env companion
test confirms all invariants hold against a production env.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import building2building.api as api_mod
from building2building.api.rl_wrappers import wrap_env_for_rl
from building2building.simulator.wrappers import NormalizeObservation
from building2building.types import RewardConfig

_FILLED_REWARD = RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0)


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


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


@pytest.mark.quick
def test_wrap_env_for_rl_real_env_invariant(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    """Companion: wrap_env_for_rl must produce correct spaces on the real env.

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
        wrapped = wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)
        np.testing.assert_array_equal(wrapped.observation_space.low, 0.0)
        np.testing.assert_array_equal(wrapped.observation_space.high, 1.0)
        np.testing.assert_allclose(wrapped.action_space.low, -1.0)
        np.testing.assert_allclose(wrapped.action_space.high, 1.0)
        assert (
            wrapped.metadata["observation_names"]
            == env.unwrapped.metadata["observation_names"]
        )
    finally:
        env.close()
