"""Audit note: temporary long marker; scheduled for T27 move to `quick/`.

Verify that the observation dimension changes depending on
``target_temperature_mode``.

In ``"constant"`` mode, occupancy and target temperature are *not* part of
the observation (they carry no information).  In ``"occupancy"`` mode they
are included (one slot each per controlled zone).

Requires EnergyPlus — run with
``B2B_RUN_LONG_TESTS=1 pytest -s tests/long/test_observation_dimension.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from building2building.config.models import DatasetSelectionConfig, EnvBuildConfig
from building2building.envs.factory import make_env_from_config
from building2building.types import RandomScheduleConfig, RewardConfig, TaskConfig

pytestmark = pytest.mark.long

N_STEPS = 4


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def _make_env(
    tmp_path: Path,
    mode: str,
    *,
    suffix: str = "",
) -> tuple:
    task = TaskConfig.from_dict(
        {
            "run_period": "winter",
            "target_temperature_mode": mode,
            "default_zone_target_temperature": {
                "occupied_c": 21.0,
                "unoccupied_c": 18.0,
            },
            **(
                {"random_schedule": {"building_type": "OfficeSmall", "seed": 0}}
                if mode == "random_schedule"
                else {}
            ),
        }
    )
    config = EnvBuildConfig(
        dataset_selection=DatasetSelectionConfig(
            building_type="OfficeSmall",
            split="train",
            mode="split_index",
            split_index=0,
        ),
        task=task,
        reward=RewardConfig(energy_weight=0.0),
        env_max_steps=N_STEPS,
    )
    env = make_env_from_config(
        config, eplus_output_dir=tmp_path / f"eplus_{mode}{suffix}"
    )
    return env


def test_constant_mode_excludes_occupancy_and_target(tmp_path: Path) -> None:
    _requires_long_runtime()

    env = _make_env(tmp_path, "constant")
    obs, _info = env.reset()

    obs_names: list[str] = env.metadata["observation_names"]
    controlled_zones: list[str] = env.metadata["controlled_zones"]
    assert len(controlled_zones) > 0

    for zone in controlled_zones:
        assert f"zone_occupancy {zone}" not in obs_names
        assert f"target_temperature {zone}" not in obs_names

    env.close()


def test_occupancy_mode_includes_occupancy_and_target(tmp_path: Path) -> None:
    _requires_long_runtime()

    env = _make_env(tmp_path, "occupancy")
    obs, _info = env.reset()

    obs_names: list[str] = env.metadata["observation_names"]
    controlled_zones: list[str] = env.metadata["controlled_zones"]
    assert len(controlled_zones) > 0

    for zone in controlled_zones:
        assert f"zone_occupancy {zone}" in obs_names
        assert f"target_temperature {zone}" in obs_names

    env.close()


def test_observation_dimension_difference(tmp_path: Path) -> None:
    """The occupancy-mode observation should have exactly 2 extra slots per
    controlled zone compared to constant mode."""
    _requires_long_runtime()

    env_const = _make_env(tmp_path, "constant", suffix="_const")
    obs_const, _ = env_const.reset()
    n_controlled = len(env_const.metadata["controlled_zones"])
    dim_const = obs_const.shape[0]
    env_const.close()

    env_occ = _make_env(tmp_path, "occupancy", suffix="_occ")
    obs_occ, _ = env_occ.reset()
    dim_occ = obs_occ.shape[0]
    env_occ.close()

    expected_diff = 2 * n_controlled
    actual_diff = dim_occ - dim_const
    assert actual_diff == expected_diff, (
        f"Expected occupancy mode to add {expected_diff} slots "
        f"(2 × {n_controlled} controlled zones), but got "
        f"dim_occ={dim_occ} - dim_const={dim_const} = {actual_diff}"
    )


def test_random_schedule_mode_includes_occupancy_and_target(tmp_path: Path) -> None:
    """``random_schedule`` must expose the same zone_occupancy /
    target_temperature slots as ``occupancy`` so that the reactive and
    PPO controllers work without modification."""
    _requires_long_runtime()

    env = _make_env(tmp_path, "random_schedule", suffix="_rs")
    obs, _info = env.reset()
    obs_names: list[str] = env.metadata["observation_names"]
    controlled_zones: list[str] = env.metadata["controlled_zones"]
    assert len(controlled_zones) > 0
    for zone in controlled_zones:
        assert f"zone_occupancy {zone}" in obs_names
        assert f"target_temperature {zone}" in obs_names
    env.close()


def test_random_schedule_dimension_matches_occupancy(tmp_path: Path) -> None:
    """The observation vector length must be identical between
    ``occupancy`` and ``random_schedule`` modes."""
    _requires_long_runtime()

    env_occ = _make_env(tmp_path, "occupancy", suffix="_occ2")
    obs_occ, _ = env_occ.reset()
    dim_occ = obs_occ.shape[0]
    env_occ.close()

    env_rs = _make_env(tmp_path, "random_schedule", suffix="_rs2")
    obs_rs, _ = env_rs.reset()
    dim_rs = obs_rs.shape[0]
    env_rs.close()

    assert dim_rs == dim_occ, (
        f"random_schedule and occupancy must have identical obs dims, "
        f"got {dim_rs} vs {dim_occ}"
    )
