"""
Unit tests for environment wrappers.
"""

import gymnasium as gym
import numpy as np
import pytest

from building2building.simulator.wrappers import (
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

    def test_augment_normalizes_params(self):
        """Test that building parameters are normalized to [-1, 1]."""
        metadata = {
            "area": 150.0,
            "warmup_phases": 3,
            "hvac_actuators": ["a1", "a2"],
        }
        env = MockEnv(metadata=metadata)
        wrapped = AugmentObservationWithBuildingParams(env, allow_defaults=True)

        # All normalized params should be in [-1, 1]
        assert np.all(wrapped.normalized_params >= -1.0)
        assert np.all(wrapped.normalized_params <= 1.0)

    def test_normalized_params_clipped_when_out_of_range(self):
        """Test that out-of-range metadata values are clipped to [-1, 1]."""
        metadata = {
            "area": 1000.0,  # Deliberately outside nominal range.
            "warmup_phases": 3,
            "hvac_actuators": ["a1"],
        }
        env = MockEnv(metadata=metadata)
        wrapped = AugmentObservationWithBuildingParams(env, allow_defaults=True)

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
        wrapped = AugmentObservationWithBuildingParams(env, allow_defaults=True)

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

    def test_zone_aware_padding_keeps_non_zone_features_consistent(self):
        """Test that non-zone features are at consistent positions across buildings.

        This is critical for multi-building generalization - outdoor temp, time
        features, and energy consumption must be at the same indices regardless
        of the number of zones.
        """
        # Building with 2 zones: obs = [zone1, zone2, outdoor_temp, outdoor_humid,
        #                                time_of_day, day_of_week, day_of_year, elec, gas]
        # = 2 zones + 7 non-zone features = 9 total
        env_2_zones = MockEnv(obs_size=9)
        wrapped_2_zones = PadObservation(env_2_zones, target_size=20)
        obs_2_zones, _ = wrapped_2_zones.reset()

        # Building with 5 zones: obs = [zone1, ..., zone5, outdoor_temp, ..., gas]
        # = 5 zones + 7 non-zone features = 12 total
        env_5_zones = MockEnv(obs_size=12)
        wrapped_5_zones = PadObservation(env_5_zones, target_size=20)
        obs_5_zones, _ = wrapped_5_zones.reset()

        # Both should be padded to 20
        assert obs_2_zones.shape[0] == 20
        assert obs_5_zones.shape[0] == 20

        # Max zones = 20 - 7 = 13
        # Non-zone features should start at index 13 for both buildings
        # For 2-zone building: [zone1, zone2, 0, 0, ..., 0 (11 padded zones), outdoor_temp, ...]
        # For 5-zone building: [zone1, ..., zone5, 0, ..., 0 (8 padded zones), outdoor_temp, ...]

        # Non-zone features should be at indices 13-19 for both
        # (outdoor_temp, outdoor_humid, time_of_day, day_of_week, day_of_year, elec, gas)
        non_zone_start_idx = 13

        # For 2-zone building: original obs[2:9] should be at padded obs[13:20]
        # For 5-zone building: original obs[5:12] should be at padded obs[13:20]
        # Both should have non-zero values at these positions
        assert np.any(obs_2_zones[non_zone_start_idx:] != 0)
        assert np.any(obs_5_zones[non_zone_start_idx:] != 0)

        # Zone padding should be zeros
        # 2-zone building: indices 2-12 should be zero (11 padded zones)
        np.testing.assert_array_equal(obs_2_zones[2:non_zone_start_idx], np.zeros(11))
        # 5-zone building: indices 5-12 should be zero (8 padded zones)
        np.testing.assert_array_equal(obs_5_zones[5:non_zone_start_idx], np.zeros(8))

    def test_pad_raises_error_if_obs_too_large(self):
        """Test that wrapper raises error if obs exceeds target size."""
        env = MockEnv(obs_size=15)

        with pytest.raises(ValueError, match="exceeds target_size"):
            PadObservation(env, target_size=10)

