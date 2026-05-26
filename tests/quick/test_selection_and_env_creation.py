"""Tests for the unified dataset selection and environment creation pipeline."""
# This file pins the public API contract.
# Changes here = breaking API changes; requires a CHANGELOG.md entry.
# Marker applied automatically by conftest.py (api_contract glob).


from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from building2building.config import (
    DatasetSelectionConfig,
    EnvBuildConfig,
    parse_benchmark_config,
)

pytestmark = pytest.mark.quick


class _DummyEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_dataset_selection_split_index_mode() -> None:
    cfg = DatasetSelectionConfig(
        building_type="OfficeSmall",
        split="train",
        mode="split_index",
        split_index=3,
    )
    assert cfg.building_type == "OfficeSmall"
    assert cfg.split_index == 3


def test_dataset_selection_building_id_mode() -> None:
    cfg = DatasetSelectionConfig(
        building_type="Warehouse",
        split="test",
        mode="building_id",
        building_id="Warehouse-0042",
    )
    assert cfg.building_id == "Warehouse-0042"


def test_parse_benchmark_config_single_type() -> None:
    parsed = parse_benchmark_config(
        {
            "mode": "single_type",
            "building_type": "OfficeSmall",
            "train": {
                "selection": {"mode": "indices", "indices": [1, 2]},
                "config": {"reward": {"reward_type": "NormalizedDeadbandRewardConfig"}},
            },
            "test": {
                "selection": {"mode": "random", "n": 2},
                "config": {"reward": {"reward_type": "NormalizedDeadbandRewardConfig"}},
            },
        }
    )
    assert parsed.mode == "single_type"


def test_env_build_config_parsing() -> None:
    cfg = EnvBuildConfig.from_dict(
        {
            "dataset_selection": {
                "building_type": "OfficeSmall",
                "split": "train",
                "mode": "split_index",
                "split_index": 0,
            },
            "task": {"run_period": "winter"},
            "reward": {"reward_type": "NormalizedDeadbandRewardConfig"},
        }
    )
    assert cfg.task.run_period.name == "winter"
    assert cfg.dataset_selection.building_type == "OfficeSmall"
