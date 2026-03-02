from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from b2b.config import DatasetSelectionConfig, EnvBuildConfig, parse_benchmark_config
from b2b.datasets.access import select_building_ids
from b2b.make_env import make_env
from b2b.types import BaseRewardConfig, BuildingConfig


pytestmark = pytest.mark.quick


class _DummyEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_dataset_selection_random_mode() -> None:
    with patch("b2b.datasets.access.sample_building_ids") as mock_sample:
        mock_sample.return_value = [11, 22]
        out = select_building_ids(
            DatasetSelectionConfig(
                dataset="single_zone_houses",
                split="train",
                mode="random",
                sample_size=2,
                seed=123,
            )
        )
    assert out == [11, 22]


def test_dataset_selection_indices_mode() -> None:
    with patch("b2b.datasets.access.building_ids_from_split_indices") as mock_indices:
        mock_indices.return_value = [101, 103]
        out = select_building_ids(
            DatasetSelectionConfig(
                dataset="multizones_reference_buildings",
                building_type="OfficeSmall",
                split="test",
                mode="split_indices",
                split_indices=[0, 2],
            )
        )
    assert out == [101, 103]


def test_parse_benchmark_config_single_type() -> None:
    parsed = parse_benchmark_config(
        {
            "mode": "single_type",
            "building_type": "OfficeSmall",
            "train": {"selection": {"mode": "indices", "indices": [1, 2]}},
            "test": {"selection": {"mode": "random", "n": 2}},
        }
    )
    assert parsed.mode == "single_type"


@patch("b2b.datasets.access.hydroquebec.search_configs")
def test_make_env_uses_typed_canonical_path(mock_search_configs: MagicMock, tmp_path: Path) -> None:
    fake_config = BuildingConfig(
        path_to_building=Path("a"),
        path_to_weather=Path("b"),
        reward_config=BaseRewardConfig(0.0),
        eplus_output_dir=Path("eplus"),
        warmup_phases=1,
        area=1.0,
        hvac_equipment=[],
    )
    mock_search_configs.return_value = [fake_config]
    with patch("b2b.envs.factory.create_simulator") as mock_create:
        mock_env = _DummyEnv()
        mock_create.return_value = mock_env
        env = make_env(
            config={
                "env": {"max_steps": 4},
                "reward": {"reward_type": "BaseRewardConfig"},
                "task": {"run_period": "winter"},
                "bldg": {
                    "dataset": "single_zone_houses",
                    "split": "train",
                    "index": 0,
                },
            },
            eplus_output_dir=tmp_path,
        )
    assert env is not None
    assert mock_create.called


def test_env_build_config_parsing() -> None:
    cfg = EnvBuildConfig.from_dict(
        {
            "dataset_selection": {
                "dataset": "single_zone_houses",
                "split": "train",
                "mode": "split_index",
                "split_index": 0,
            },
            "task": {"run_period": "winter"},
            "reward": {"reward_type": "BarrierRewardConfig"},
        }
    )
    assert cfg.task.run_period.name == "winter"
