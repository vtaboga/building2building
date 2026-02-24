from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from b2b.make_env import make_env


pytestmark = pytest.mark.long


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def test_single_zone_pipeline_selection_and_env_creation(tmp_path: Path) -> None:
    _requires_long_runtime()
    cfg = OmegaConf.create(
        {
            "env": {"max_steps": 8, "normalize_obs": False},
            "reward": {"reward_type": "BaseRewardConfig"},
            "task": {"run_period": "winter"},
            "bldg": {
                "selection": {"enabled": True, "split": "train", "index": 0},
                "bldg": {},
            },
        }
    )
    env = make_env(config=cfg, eplus_output_dir=tmp_path / "eplus_outputs")
    try:
        obs, _ = env.reset()
        assert obs is not None
        assert hasattr(env, "action_space")
        assert env.action_space.shape[0] > 0
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not (terminated and truncated)
    finally:
        env.close()
