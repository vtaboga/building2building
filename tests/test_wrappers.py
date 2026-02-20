"""
Unit tests for environment wrappers.
"""

import gymnasium as gym
import numpy as np
import pytest

from b2b.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
    PadObservation,
)


class MockEnv(gym.Env):
    """Mock environment for testing wrappers."""

    def __init__(self, obs_size=10, action_size=3, metadata=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32
        )
        self.metadata = metadata or {}
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        obs = np.random.uniform(-5, 5, size=self.observation_space.shape).astype(
            np.float32
        )
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.uniform(-5, 5, size=self.observation_space.shape).astype(
            np.float32
        )
        reward = 0.0
        terminated = self._step_count >= 10
        truncated = False
        return obs, reward, terminated, truncated, {}


class TestAugmentObservationWithBuildingParams:
    """Tests for AugmentObservationWithBuildingParams wrapper."""

    def test_augment_increases_observation_size(self):
        """Test that wrapper increases observation space size."""
        metadata = {
            "area": 150.0,
            "warmup_phases": 3,
            "hvac_actuators": ["a1", "a2"],
        }
        env = MockEnv(obs_size=10, metadata=metadata)
        wrapped = AugmentObservationWithBuildingParams(env)

        # Should add 5 parameters: area, warmup_phases, num_actuators, year_built, num_units
        assert wrapped.observation_space.shape[0] == 10 + 5

    def test_augment_extracts_from_metadata(self):
        """Test that wrapper extracts building params from metadata."""
        metadata = {
            "area": 200.0,
            "warmup_phases": 4,
            "hvac_actuators": ["a1", "a2", "a3"],
        }
        env = MockEnv(metadata=metadata)
        wrapped = AugmentObservationWithBuildingParams(env)

        assert wrapped.building_params["area"] == 200.0
        assert wrapped.building_params["warmup_phases"] == 4.0
        assert wrapped.building_params["num_actuators"] == 3.0

    def test_augment_uses_defaults_when_missing(self):
        """Test that wrapper uses defaults for missing parameters."""
        env = MockEnv(metadata={})
        wrapped = AugmentObservationWithBuildingParams(env)

        # Should use defaults
        assert "area" in wrapped.building_params
        assert "warmup_phases" in wrapped.building_params
        assert "num_actuators" in wrapped.building_params

    def test_augment_normalizes_params(self):
        """Test that building parameters are normalized to [-1, 1]."""
        metadata = {
            "area": 150.0,
            "warmup_phases": 3,
            "hvac_actuators": ["a1", "a2"],
        }
        env = MockEnv(metadata=metadata)
        wrapped = AugmentObservationWithBuildingParams(env)

        # All normalized params should be in [-1, 1]
        assert np.all(wrapped.normalized_params >= -1.0)
        assert np.all(wrapped.normalized_params <= 1.0)

    def test_augment_appends_to_observation(self):
        """Test that wrapper appends params to observations."""
        metadata = {
            "area": 150.0,
            "warmup_phases": 3,
            "hvac_actuators": ["a1", "a2"],
        }
        env = MockEnv(obs_size=10, metadata=metadata)
        wrapped = AugmentObservationWithBuildingParams(env)

        obs, _ = wrapped.reset()

        # Check that observation has correct size
        assert obs.shape[0] == 15  # 10 original + 5 params

        # Check that last 5 elements are the normalized params
        np.testing.assert_array_equal(obs[-5:], wrapped.normalized_params)

    def test_augment_with_custom_params(self):
        """Test that wrapper accepts custom building params."""
        env = MockEnv(obs_size=10)
        custom_params = {
            "area": 250.0,
            "warmup_phases": 5,
            "num_actuators": 4.0,
            "year_built": 2000.0,
            "num_units": 2.0,
        }
        wrapped = AugmentObservationWithBuildingParams(
            env, building_params=custom_params
        )

        assert wrapped.building_params == custom_params


class TestNormalizeObservation:
    """Tests for NormalizeObservation wrapper."""

    def test_normalize_scales_to_zero_to_one(self):
        """Test that observations are normalized to [0, 1]."""
        env = MockEnv(obs_size=5)
        wrapped = NormalizeObservation(env)

        obs, _ = wrapped.reset()

        # All observations should be in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_normalize_handles_edge_values(self):
        """Test normalization at observation space boundaries."""
        env = MockEnv(obs_size=5)
        wrapped = NormalizeObservation(env)

        # Test with min values
        obs_min = env.observation_space.low
        normalized_min = wrapped.observation(obs_min)
        np.testing.assert_array_almost_equal(normalized_min, np.zeros(5))

        # Test with max values
        obs_max = env.observation_space.high
        normalized_max = wrapped.observation(obs_max)
        np.testing.assert_array_almost_equal(normalized_max, np.ones(5))


class TestPadObservation:
    """Tests for PadObservation wrapper."""

    def test_pad_increases_observation_size(self):
        """Test that wrapper pads observations to target size."""
        env = MockEnv(obs_size=8)
        wrapped = PadObservation(env, target_size=12)

        assert wrapped.observation_space.shape[0] == 12

    def test_pad_preserves_original_values(self):
        """Test that original observation values are preserved."""
        env = MockEnv(obs_size=8)
        wrapped = PadObservation(env, target_size=12)

        obs, _ = wrapped.reset()

        # First 8 values should be non-zero (from env)
        # Last 4 values should be zero (padding)
        assert obs.shape[0] == 12
        np.testing.assert_array_equal(obs[8:], np.zeros(4))

    def test_pad_raises_error_if_obs_too_large(self):
        """Test that wrapper raises error if obs exceeds target size."""
        env = MockEnv(obs_size=15)

        with pytest.raises(ValueError, match="exceeds target_size"):
            PadObservation(env, target_size=10)
