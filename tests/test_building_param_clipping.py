"""Test building parameter clipping warnings."""

import logging
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from b2b.simulator.wrappers import AugmentObservationWithBuildingParams


@pytest.fixture
def mock_env():
    """Create a mock environment with observation space."""
    env = MagicMock(spec=gym.Env)
    env.observation_space = gym.spaces.Box(
        low=np.array([0.0, 0.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32,
    )
    env.metadata = {
        "area": 100.0,
        "warmup_phases": 3.0,
        "hvac_actuators": [MagicMock()],
    }
    return env


def test_no_clipping_within_range(mock_env, caplog):  # noqa: ARG001
    """Test that no warnings are logged when values are within range."""
    with caplog.at_level(logging.WARNING):
        _ = AugmentObservationWithBuildingParams(mock_env)
        # All values should be within range
        assert len(caplog.records) == 0


def test_clipping_area_too_large(mock_env, caplog):  # noqa: ARG001
    """Test that warning is logged when area exceeds max range."""
    mock_env.metadata["area"] = 600.0  # Exceeds max of 500.0

    with caplog.at_level(logging.WARNING):
        _ = AugmentObservationWithBuildingParams(mock_env)

    # Should have warning about area being clipped
    assert any("area" in record.message.lower() for record in caplog.records)
    assert any("clipped" in record.message.lower() for record in caplog.records)


def test_clipping_area_too_small(mock_env, caplog):  # noqa: ARG001
    """Test that warning is logged when area is below min range."""
    mock_env.metadata["area"] = 30.0  # Below min of 50.0

    with caplog.at_level(logging.WARNING):
        _ = AugmentObservationWithBuildingParams(mock_env)

    # Should have warning about area being clipped
    assert any("area" in record.message.lower() for record in caplog.records)
    assert any("clipped" in record.message.lower() for record in caplog.records)


def test_clipping_num_units_too_large(mock_env, caplog):  # noqa: ARG001
    """Test that warning is logged when num_units exceeds max range."""
    mock_env.metadata["building_source_metadata"] = {
        "geometry_building_num_units": 15.0  # Exceeds max of 10.0
    }

    with caplog.at_level(logging.WARNING):
        _ = AugmentObservationWithBuildingParams(mock_env)

    # Should have warning about num_units being clipped
    assert any("num_units" in record.message.lower() for record in caplog.records)
    assert any("clipped" in record.message.lower() for record in caplog.records)


@patch("b2b.simulator.wrappers.wandb")
def test_wandb_logging_on_clipping(mock_wandb, mock_env):  # noqa: ARG001
    """Test that clipped parameters are logged to Wandb."""
    mock_wandb.run = MagicMock()
    mock_env.metadata["area"] = 600.0  # Will be clipped

    _ = AugmentObservationWithBuildingParams(mock_env)

    # Wandb.log should have been called with clipping info
    assert mock_wandb.log.called
    calls = mock_wandb.log.call_args_list

    # Check that we logged the clipped area value
    logged_data = {}
    for call in calls:
        logged_data.update(call[0][0])

    assert "building_params/clipped_area_value" in logged_data
    assert logged_data["building_params/clipped_area_value"] == 600.0
    assert "building_params/num_clipped_params" in logged_data
    assert logged_data["building_params/num_clipped_params"] >= 1


@patch("b2b.simulator.wrappers.wandb")
def test_wandb_logging_on_reset(mock_wandb, mock_env):  # noqa: ARG001
    """Test that building parameters are logged to Wandb on reset."""
    mock_wandb.run = MagicMock()
    mock_env.reset.return_value = (np.array([0.5, 0.5, 0.5]), {})

    wrapper = AugmentObservationWithBuildingParams(mock_env)
    wrapper.reset()

    # Wandb.log should have been called with building params
    assert mock_wandb.log.called
    calls = mock_wandb.log.call_args_list

    # Check that we logged building parameter values
    logged_data = {}
    for call in calls:
        logged_data.update(call[0][0])

    assert "building_params/area" in logged_data
    assert "building_params/warmup_phases" in logged_data
    assert "building_params/num_actuators" in logged_data


def test_normalized_values_are_clipped(mock_env):  # noqa: ARG001
    """Test that normalized values are properly clipped to [-1, 1]."""
    mock_env.metadata["area"] = 1000.0  # Way outside range

    wrapper = AugmentObservationWithBuildingParams(mock_env)

    # All normalized params should be in [-1, 1]
    assert np.all(wrapper.normalized_params >= -1.0)
    assert np.all(wrapper.normalized_params <= 1.0)
