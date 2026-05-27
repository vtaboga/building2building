"""Audit note: temporary long marker; scheduled for T27 deletion."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import building2building as b2b

pytestmark = pytest.mark.long


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def test_single_family_house_env_creation(tmp_path: Path) -> None:
    _requires_long_runtime()
    env = b2b.new_make_env(
        "SingleFamilyHouse",
        split="train",
        index=0,
        task="task1",
        eplus_output_dir=tmp_path / "eplus_outputs",
        max_episode_steps=8,
    )
    try:
        obs, _ = env.reset()
        assert obs is not None
        assert hasattr(env, "action_space")
        assert env.action_space.shape[0] > 0
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not (terminated and truncated)
    finally:
        env.close()
