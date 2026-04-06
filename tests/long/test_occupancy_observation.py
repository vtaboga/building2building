"""Verify that zone occupancy observations are available and time-varying
for an OfficeSmall building, and that the target temperature responds to
occupancy when ``target_temperature_mode`` is ``"occupancy"``.

Requires EnergyPlus — run with ``B2B_RUN_LONG_TESTS=1 pytest -s tests/long/test_occupancy_observation.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from building2building.config.models import DatasetSelectionConfig, EnvBuildConfig
from building2building.envs.factory import make_env_from_config
from building2building.types import BaseRewardConfig, TaskConfig


pytestmark = pytest.mark.long

OCCUPIED_C = 22.0
UNOCCUPIED_C = 18.0
N_STEPS = 200  # ~2 days at 4 timesteps/hour


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def _make_office_small_env(tmp_path: Path) -> tuple:
    task = TaskConfig.from_dict(
        {
            "run_period": "winter",
            "target_temperature_mode": "occupancy",
            "default_zone_target_temperature": {
                "occupied_c": OCCUPIED_C,
                "unoccupied_c": UNOCCUPIED_C,
            },
        }
    )
    config = EnvBuildConfig(
        dataset_selection=DatasetSelectionConfig(
            dataset="multizones_reference_buildings",
            building_type="OfficeSmall",
            split="train",
            mode="split_index",
            split_index=0,
        ),
        task=task,
        reward=BaseRewardConfig(energy_weight=0.0),
        env_max_steps=N_STEPS,
    )
    env = make_env_from_config(config, eplus_output_dir=tmp_path / "eplus")
    return env


def test_occupancy_varies_and_drives_target_temperature(tmp_path: Path) -> None:
    _requires_long_runtime()

    env = _make_office_small_env(tmp_path)
    obs, info = env.reset()

    obs_names: list[str] = env.metadata["observation_names"]
    controlled_zones: list[str] = env.metadata["controlled_zones"]

    assert len(controlled_zones) > 0, "No controlled zones found"

    occupancy_indices: dict[str, int] = {}
    target_indices: dict[str, int] = {}
    for zone in controlled_zones:
        occ_name = f"zone_occupancy {zone}"
        tgt_name = f"target_temperature {zone}"
        assert occ_name in obs_names, f"Missing occupancy slot for {zone}"
        assert tgt_name in obs_names, f"Missing target_temperature slot for {zone}"
        occupancy_indices[zone] = obs_names.index(occ_name)
        target_indices[zone] = obs_names.index(tgt_name)

    occupancy_trace: dict[str, list[float]] = {z: [] for z in controlled_zones}
    target_trace: dict[str, list[float]] = {z: [] for z in controlled_zones}

    for _ in range(N_STEPS):
        action = env.action_space.sample() * 0.0  # zero action
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

        for zone in controlled_zones:
            occupancy_trace[zone].append(float(obs[occupancy_indices[zone]]))
            target_trace[zone].append(float(obs[target_indices[zone]]))

    print("\n--- Occupancy & Target Temperature Summary ---")
    any_zone_varies = False
    for zone in controlled_zones:
        occ = np.array(occupancy_trace[zone])
        tgt = np.array(target_trace[zone])
        n_occupied = int(np.count_nonzero(occ > 0))
        n_unoccupied = int(np.count_nonzero(occ == 0))
        unique_targets = np.unique(tgt)

        print(f"  Zone: {zone}")
        print(f"    Occupancy: min={occ.min():.1f}  max={occ.max():.1f}  "
              f"occupied_steps={n_occupied}  unoccupied_steps={n_unoccupied}")
        print(f"    Target temps: {sorted(unique_targets)}")

        if n_occupied > 0 and n_unoccupied > 0:
            any_zone_varies = True
            assert OCCUPIED_C in unique_targets, (
                f"Zone {zone}: expected {OCCUPIED_C}°C in targets when occupied"
            )
            assert UNOCCUPIED_C in unique_targets, (
                f"Zone {zone}: expected {UNOCCUPIED_C}°C in targets when unoccupied"
            )

    assert any_zone_varies, (
        "Expected at least one zone with both occupied and unoccupied timesteps "
        f"over {N_STEPS} steps (~2 days). Zones: {controlled_zones}"
    )

    env.close()
    print("--- PASS ---")
