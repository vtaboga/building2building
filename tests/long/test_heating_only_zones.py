"""Test heating-only zones support with the G36 baseline on a Warehouse building.

Runs the UnitaryG36Policy on a Warehouse from the test split under both
``expose_heating_only_zones`` settings and verifies:

- heating-only zone metadata is populated
- action space dimensions change correctly
- zone temperatures stay within physical bounds
- the policy completes without errors

Requires EnergyPlus — run with
``B2B_RUN_LONG_TESTS=1 pytest -s tests/long/test_heating_only_zones.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy
from b2b.benchmark.runner import RolloutData, run_rollout
from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig
from b2b.envs.factory import make_env_from_config
from b2b.types import BaseRewardConfig, TaskConfig


pytestmark = pytest.mark.long

N_STEPS = 96


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


WAREHOUSE_G36_CFG = OmegaConf.create(
    {
        "heating_setpoint_c": 20.0,
        "cooling_setpoint_c": 22.0,
        "kp": 0.25,
        "ki": 0.02,
        "integral_max": 200.0,
        "min_fan_fraction": 0.15,
        "sat_min_c": 12.0,
        "sat_max_c": 35.0,
        "sat_initial_c": 21.0,
        "sat_trim": 0.5,
        "sat_respond": 1.0,
        "demand_deadband": 0.3,
        "availability_on": 2.0,
        "target_schedule": {"enabled": False},
    }
)


def _make_warehouse_env(
    tmp_path: Path,
    expose_heating_only: bool,
) -> Any:
    task = TaskConfig.from_dict({"run_period": "winter"})
    suffix = "exposed" if expose_heating_only else "hidden"
    config = EnvBuildConfig(
        dataset_selection=DatasetSelectionConfig(
            dataset="multizones_reference_buildings",
            building_type="Warehouse",
            split="test",
            mode="split_index",
            split_index=0,
        ),
        task=task,
        reward=BaseRewardConfig(energy_weight=0.0),
        env_max_steps=N_STEPS,
        expose_heating_only_zones=expose_heating_only,
    )
    return make_env_from_config(
        config, eplus_output_dir=tmp_path / f"eplus_warehouse_{suffix}"
    )


def _extract_zone_temps(
    obs: np.ndarray,
    obs_names: list[str],
    zones: list[str],
) -> dict[str, np.ndarray]:
    """Extract zone temperature time series from the flat observation matrix."""
    prefix = "zone air temperature"
    out: dict[str, np.ndarray] = {}
    for zone in zones:
        zn = zone.strip().lower()
        for i, name in enumerate(obs_names):
            slot = str(name).strip().lower()
            if not slot.startswith(prefix):
                continue
            zone_part = slot[len(prefix) :].strip()
            if zone_part == zn or zn in zone_part or zone_part in zn:
                out[zone] = obs[:, i]
                break
    return out


def _run_warehouse_rollout(
    tmp_path: Path,
    expose_heating_only: bool,
) -> tuple[Any, RolloutData]:
    """Create a Warehouse env, run one G36 episode, return env + data."""
    env = _make_warehouse_env(tmp_path, expose_heating_only)
    policy = UnitaryG36Policy(WAREHOUSE_G36_CFG)

    results, data = run_rollout(
        env=env,
        policy=policy,
        n_episodes=1,
        deterministic=True,
        max_steps=N_STEPS,
        record=True,
    )

    assert len(results) == 1
    assert results[0].n_steps == N_STEPS
    assert data is not None

    return env, data


# -- tests -----------------------------------------------------------------


def test_warehouse_g36_exposed(tmp_path: Path) -> None:
    """G36 on a Warehouse with heating-only zones exposed to the agent."""
    _requires_long_runtime()

    env, data = _run_warehouse_rollout(tmp_path, expose_heating_only=True)
    try:
        controlled_zones: list[str] = env.metadata["controlled_zones"]
        heating_only: list[str] = env.metadata["heating_only_zones"]
        obs_names: list[str] = env.metadata["observation_names"]
        action_names: list[str] = env.metadata["action_names"]

        assert len(heating_only) > 0, (
            "Warehouse should have heating-only zones"
        )
        assert all(z in controlled_zones for z in heating_only), (
            "heating-only zones must be a subset of controlled zones"
        )

        htg_sp_actions = [
            n for n in action_names if "heating only htg setpoint" in n.lower()
        ]
        assert len(htg_sp_actions) == len(heating_only), (
            f"Expected {len(heating_only)} heating-only zone setpoint "
            f"actuators in agent action space, got {len(htg_sp_actions)}"
        )

        zone_temps = _extract_zone_temps(data.obs, obs_names, controlled_zones)
        for zone in controlled_zones:
            assert zone in zone_temps, (
                f"Could not find temperature observation for zone {zone}"
            )
            temps = zone_temps[zone]
            assert np.all(temps > -10.0) and np.all(temps < 55.0), (
                f"Zone {zone!r} temperatures out of physical bounds: "
                f"min={temps.min():.1f}, max={temps.max():.1f}"
            )
    finally:
        env.close()


def test_warehouse_g36_hidden(tmp_path: Path) -> None:
    """G36 on a Warehouse with heating-only zones pinned (not exposed)."""
    _requires_long_runtime()

    env, data = _run_warehouse_rollout(tmp_path, expose_heating_only=False)
    try:
        controlled_zones: list[str] = env.metadata["controlled_zones"]
        heating_only: list[str] = env.metadata["heating_only_zones"]
        obs_names: list[str] = env.metadata["observation_names"]
        action_names: list[str] = env.metadata["action_names"]

        assert len(heating_only) > 0, (
            "Warehouse should still report heating-only zones in metadata"
        )
        assert all(z in controlled_zones for z in heating_only), (
            "heating-only zones must still be in controlled_zones"
        )

        htg_sp_actions = [
            n for n in action_names if "heating only htg setpoint" in n.lower()
        ]
        assert len(htg_sp_actions) == 0, (
            f"Heating-only zone setpoint actuators should be hidden "
            f"from the agent, but found {len(htg_sp_actions)}"
        )

        zone_temps = _extract_zone_temps(data.obs, obs_names, controlled_zones)
        for zone in controlled_zones:
            assert zone in zone_temps, (
                f"Could not find temperature observation for zone {zone}"
            )
            temps = zone_temps[zone]
            assert np.all(temps > -10.0) and np.all(temps < 55.0), (
                f"Zone {zone!r} temperatures out of physical bounds: "
                f"min={temps.min():.1f}, max={temps.max():.1f}"
            )
    finally:
        env.close()


def test_action_space_dimension_difference(tmp_path: Path) -> None:
    """Exposed mode should have more action dims than hidden mode —
    exactly the number of heating-only zone setpoint actuators."""
    _requires_long_runtime()

    env_exposed = _make_warehouse_env(
        tmp_path / "dim_check", expose_heating_only=True
    )
    try:
        n_act_exposed = env_exposed.action_space.shape[0]
        n_heating_only = len(env_exposed.metadata["heating_only_zones"])
    finally:
        env_exposed.close()

    env_hidden = _make_warehouse_env(
        tmp_path / "dim_check", expose_heating_only=False
    )
    try:
        n_act_hidden = env_hidden.action_space.shape[0]
    finally:
        env_hidden.close()

    assert n_act_exposed - n_act_hidden == n_heating_only, (
        f"Expected action space difference of {n_heating_only} "
        f"(heating-only zones), got "
        f"{n_act_exposed} - {n_act_hidden} = {n_act_exposed - n_act_hidden}"
    )
