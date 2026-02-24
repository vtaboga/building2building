from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from b2b.benchmark.runner import EpisodeResult
from b2b.benchmark.rollout_multizones import ALL_BUILDING_TYPES, run_multizones_rollout


pytestmark = pytest.mark.quick


class _DummyEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.metadata = {"action_names": ["a0"], "controlled_zones": ["z0"]}

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class _DummyPolicy:
    def predict(self, obs, deterministic: bool = True):
        return np.zeros(1, dtype=np.float32), None


@pytest.mark.parametrize("building_type", ALL_BUILDING_TYPES)
@patch("b2b.benchmark.rollout_multizones.run_rollout")
@patch("b2b.benchmark.rollout_multizones.make_env")
@patch("b2b.benchmark.rollout_multizones.select_buildings")
def test_baseline_rollout_executes_for_each_building_type(
    mock_select: MagicMock,
    mock_make_env: MagicMock,
    mock_rollout: MagicMock,
    tmp_path: Path,
    building_type: str,
) -> None:
    mock_select.return_value = [
        {
            "building_id": 1,
            "building_type": building_type,
            "place": "mock_place",
            "weather_file": "weather.epw",
            "derivation_thunk": lambda: None,
        }
    ]
    mock_env = _DummyEnv()
    mock_make_env.return_value = mock_env
    fake_episode = EpisodeResult(total_reward=1.0, n_steps=4)
    mock_rollout.return_value = ([fake_episode], {})

    cfg = OmegaConf.create(
        {
            "policy": {
                "type": "custom",
                "module": "tests.quick.test_baseline_execution",
                "class_name": "_DummyPolicy",
            },
            "multizones": {"types": [building_type], "n_per_type": 1},
            "env": {"max_steps": 4},
            "task": {"run_period": "winter"},
            "reward": {"reward_type": "BaseRewardConfig"},
        }
    )
    records = run_multizones_rollout(cfg, output_dir=tmp_path)
    assert len(records) == 1
    assert records[0].success is True
