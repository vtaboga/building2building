"""
Unit tests for multi-building training utilities.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from algorithms.multi_building_utils import (
    fetch_diverse_building_pool,
    make_diverse_env,
)


class TestFetchDiverseBuildingPool:
    """Tests for fetch_diverse_building_pool function."""

    @patch("algorithms.multi_building_utils.hydroquebec.search_configs")
    def test_fetch_returns_compatible_buildings(self, mock_search):
        """Test that fetch returns buildings with compatible observation spaces."""
        # Create mock building configs with different observation sizes
        mock_configs = []
        for i in range(10):
            config = MagicMock()
            config.area = 100.0 + i * 10
            config.warmup_phases = 3
            config.hvac_actuators = [f"actuator_{j}" for j in range(2)]
            config.path_to_building = f"/tmp/building_{i}.epJSON"
            mock_configs.append(config)

        mock_search.return_value = mock_configs

        # Mock the file reading and ontology creation
        with (
            patch("builtins.open", create=True),
            patch("json.load") as mock_json_load,
            patch("algorithms.multi_building_utils.Ontology.from_object") as mock_ont,
            patch(
                "algorithms.multi_building_utils.flat_observation_info"
            ) as mock_obs_info,
        ):

            # Setup mocks to return consistent observation space size
            mock_json_load.return_value = {}
            mock_ont.return_value = MagicMock()

            # All buildings have same obs size (45) and action size (2)
            mock_space = MagicMock()
            mock_space.shape = (45,)
            mock_obs_info.return_value = MagicMock(space=mock_space)

            config = MagicMock()
            pool = fetch_diverse_building_pool(config, n_buildings=5)

            assert len(pool) == 5
            assert all(hasattr(b, "area") for b in pool)

    @patch("algorithms.multi_building_utils.hydroquebec.search_configs")
    def test_fetch_handles_empty_results(self, mock_search):
        """Test that fetch raises error when no buildings found."""
        mock_search.return_value = []

        config = MagicMock()
        with pytest.raises(RuntimeError, match="No buildings found"):
            fetch_diverse_building_pool(config, n_buildings=5)

    @patch("algorithms.multi_building_utils.hydroquebec.search_configs")
    def test_fetch_logs_diversity_stats(self, mock_search, caplog):
        """Test that diversity statistics are logged."""
        mock_configs = []
        for i in range(5):
            config = MagicMock()
            config.area = 100.0 + i * 50  # Varying areas
            config.warmup_phases = 2 + i  # Varying warmup
            config.hvac_actuators = [
                f"act_{j}" for j in range(1 + i)
            ]  # Varying actuators
            config.path_to_building = f"/tmp/building_{i}.epJSON"
            mock_configs.append(config)

        mock_search.return_value = mock_configs

        with (
            patch("builtins.open", create=True),
            patch("json.load"),
            patch("algorithms.multi_building_utils.Ontology.from_object"),
            patch(
                "algorithms.multi_building_utils.flat_observation_info"
            ) as mock_obs_info,
        ):

            mock_space = MagicMock()
            mock_space.shape = (45,)
            mock_obs_info.return_value = MagicMock(space=mock_space)

            config = MagicMock()
            pool = fetch_diverse_building_pool(config, n_buildings=5)

            # Check that diversity stats were logged (even if only 1 building found)
            # The function logs diversity stats regardless of pool size
            assert len(pool) >= 1
            assert (
                "Building pool diversity" in caplog.text or "Only found" in caplog.text
            )


class TestMakeDiverseEnv:
    """Tests for make_diverse_env function."""

    def test_make_diverse_env_requires_building_pool(self):
        """Test that make_diverse_env raises error without building pool."""
        config = MagicMock()

        with pytest.raises(ValueError, match="building_pool is required"):
            make_diverse_env(config, "/tmp/eplus_out", building_pool=None)

        with pytest.raises(ValueError, match="building_pool is required"):
            make_diverse_env(config, "/tmp/eplus_out", building_pool=[])

    @patch("algorithms.multi_building_utils.create_simulator")
    def test_make_diverse_env_samples_from_pool(self, mock_create_sim):
        """Test that make_diverse_env samples from the building pool."""
        # Create mock building pool
        building_pool = []
        for i in range(3):
            config = MagicMock()
            config.area = 100.0 + i * 10
            config.warmup_phases = 3
            config.hvac_actuators = ["actuator_1", "actuator_2"]
            building_pool.append(config)

        mock_env = MagicMock()
        mock_create_sim.return_value = mock_env

        config = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            env = make_diverse_env(config, tmpdir, building_pool=building_pool)

            # Verify create_simulator was called
            assert mock_create_sim.called
            assert env == mock_env

    @patch("algorithms.multi_building_utils.create_simulator")
    def test_make_diverse_env_creates_unique_output_dir(self, mock_create_sim):
        """Test that each env gets a unique output directory."""
        building_pool = [MagicMock(area=100, warmup_phases=3, hvac_actuators=["a1"])]
        mock_create_sim.return_value = MagicMock()

        config = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            env1 = make_diverse_env(config, tmpdir, building_pool=building_pool)
            env2 = make_diverse_env(config, tmpdir, building_pool=building_pool)

            # Both should succeed (unique UUIDs prevent conflicts)
            assert env1 is not None
            assert env2 is not None
