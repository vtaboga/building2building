"""Verify that ``gymnasium.wrappers.RescaleAction`` correctly rescales the
action space to [-1, 1] while preserving environment functionality.

Tests both a single-zone house and a multizones OfficeSmall building,
with and without the wrapper applied.

Requires EnergyPlus — run with
``B2B_RUN_LONG_TESTS=1 pytest -s tests/long/test_rescale_action.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from building2building.config.models import DatasetSelectionConfig, EnvBuildConfig
from building2building.envs.factory import make_env_from_config
from building2building.types import BaseRewardConfig, TaskConfig


pytestmark = pytest.mark.long

N_STEPS = 4


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def _make_office_env(tmp_path: Path, suffix: str = "") -> gym.Env:
    config = EnvBuildConfig(
        dataset_selection=DatasetSelectionConfig(
            dataset="multizones_reference_buildings",
            building_type="OfficeSmall",
            split="train",
            mode="split_index",
            split_index=0,
        ),
        task=TaskConfig.from_dict({"run_period": "winter"}),
        reward=BaseRewardConfig(energy_weight=0.0),
        env_max_steps=N_STEPS,
    )
    return make_env_from_config(
        config, eplus_output_dir=tmp_path / f"eplus_office{suffix}"
    )


def _make_single_zone_env(tmp_path: Path, suffix: str = "") -> gym.Env:
    config = EnvBuildConfig(
        dataset_selection=DatasetSelectionConfig(
            dataset="single_zone_houses",
            split="train",
            mode="split_index",
            split_index=0,
        ),
        task=TaskConfig.from_dict({"run_period": "winter"}),
        reward=BaseRewardConfig(energy_weight=0.0),
        env_max_steps=N_STEPS,
    )
    return make_env_from_config(
        config, eplus_output_dir=tmp_path / f"eplus_single{suffix}"
    )


@pytest.mark.parametrize("make_env_fn", ["office", "single_zone"])
def test_raw_action_space_has_physical_bounds(
    tmp_path: Path, make_env_fn: str
) -> None:
    """Without RescaleAction the action space bounds match the physical actuator
    ranges (which are NOT [-1, 1])."""
    _requires_long_runtime()

    env = (
        _make_office_env(tmp_path, "_raw")
        if make_env_fn == "office"
        else _make_single_zone_env(tmp_path, "_raw")
    )
    low, high = env.action_space.low, env.action_space.high

    assert not (
        np.allclose(low, -1.0) and np.allclose(high, 1.0)
    ), "Raw env should NOT already have [-1, 1] action bounds"

    env.close()


@pytest.mark.parametrize("make_env_fn", ["office", "single_zone"])
def test_rescale_action_normalizes_bounds(
    tmp_path: Path, make_env_fn: str
) -> None:
    """With RescaleAction the agent-facing action space should be [-1, 1]."""
    _requires_long_runtime()

    env = (
        _make_office_env(tmp_path, "_rescaled")
        if make_env_fn == "office"
        else _make_single_zone_env(tmp_path, "_rescaled")
    )
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)

    assert np.allclose(env.action_space.low, -1.0), (
        f"Expected all lows to be -1.0, got {env.action_space.low}"
    )
    assert np.allclose(env.action_space.high, 1.0), (
        f"Expected all highs to be 1.0, got {env.action_space.high}"
    )

    env.close()


@pytest.mark.parametrize("make_env_fn", ["office", "single_zone"])
def test_rescale_action_steps_successfully(
    tmp_path: Path, make_env_fn: str
) -> None:
    """The wrapped env should accept actions in [-1, 1] and step without error."""
    _requires_long_runtime()

    env = (
        _make_office_env(tmp_path, "_step")
        if make_env_fn == "office"
        else _make_single_zone_env(tmp_path, "_step")
    )
    raw_space = env.action_space
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)

    obs, _info = env.reset()
    assert obs is not None

    for _ in range(N_STEPS):
        action = env.action_space.sample()
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        if terminated or truncated:
            break

    env.close()


@pytest.mark.parametrize("make_env_fn", ["office", "single_zone"])
def test_action_dimension_unchanged(
    tmp_path: Path, make_env_fn: str
) -> None:
    """RescaleAction should not change the number of action dimensions."""
    _requires_long_runtime()

    env = (
        _make_office_env(tmp_path, "_dim")
        if make_env_fn == "office"
        else _make_single_zone_env(tmp_path, "_dim")
    )
    raw_shape = env.action_space.shape
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)

    assert env.action_space.shape == raw_shape, (
        f"Expected shape {raw_shape}, got {env.action_space.shape}"
    )

    env.close()
