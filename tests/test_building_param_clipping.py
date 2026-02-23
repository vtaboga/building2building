"""Test building parameter clipping warnings."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

from b2b.simulator.wrappers import AugmentObservationWithBuildingParams


def test_normalized_values_are_clipped():
    """Test that normalized values are properly clipped to [-1, 1]."""
    mock_env = MagicMock(spec=gym.Env)
    mock_env.observation_space = gym.spaces.Box(
        low=np.array([0.0, 0.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32,
    )
    mock_env.metadata = {
        "area": 1000.0,  # Way outside range - will be clipped
        "warmup_phases": 3.0,
        "hvac_actuators": [MagicMock()],
    }

    wrapper = AugmentObservationWithBuildingParams(mock_env)

    # All normalized params should be in [-1, 1]
    assert np.all(wrapper.normalized_params >= -1.0)
    assert np.all(wrapper.normalized_params <= 1.0)
