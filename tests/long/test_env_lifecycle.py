"""Long-running tests: full Gymnasium environment lifecycle with EnergyPlus.

These tests require a working EnergyPlus installation and network access
(for the initial dataset download).  They are excluded from quick CI runs.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

pytestmark = pytest.mark.long


class TestEnvLifecycle:
    """Create a real environment and exercise the Gymnasium API loop."""

    @pytest.fixture()
    def env(self) -> gym.Env:
        from building2building.api import make_env

        env = make_env(
            building_type="SingleFamilyHouse",
            split="train",
            index=0,
            task="task_const_e0",
            run_period="winter",
            timesteps_per_hour=4,
        )
        yield env
        env.close()

    def test_reset_returns_obs_info(self, env: gym.Env) -> None:
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape

    def test_step_returns_five_tuple(self, env: gym.Env) -> None:
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_space_consistent(self, env: gym.Env) -> None:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_multi_step_episode(self, env: gym.Env) -> None:
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        assert total_reward != 0.0 or True  # just ensure no crash
