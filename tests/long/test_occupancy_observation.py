"""Verify that zone occupancy observations are available and time-varying
for an OfficeSmall building, and that the target temperature responds to
occupancy when ``target_temperature_mode`` is ``"occupancy"``.

Built through the public ``make_env`` entry point so the requested
``run_period`` is actually applied to the simulation (the lower-level
``make_env_from_config`` does not patch the run period — see
``building2building.api.make_env`` / ``_patch_epjson_run_period``).

Requires EnergyPlus — run with ``B2B_RUN_LONG_TESTS=1 pytest -s tests/long/test_occupancy_observation.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import building2building as b2b

pytestmark = pytest.mark.long

# Values come from the ``task_occ_e0`` preset (occupancy mode, seasonal
# unoccupied policy).  Occupied setpoint is 21 C; the unoccupied setpoint
# in the winter run period is the seasonal-winter value, 18 C.
OCCUPIED_C = 21.0
UNOCCUPIED_C = 18.0
TIMESTEPS_PER_HOUR = 12
# The winter run period starts Jan 1 (a Sunday + New Year holiday), so an
# office is unoccupied for the whole first day.  Roll out 4 days to reach
# weekday daytime occupancy (the first occupied step is ~06:00 on Jan 2).
N_STEPS = TIMESTEPS_PER_HOUR * 24 * 4


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def _make_office_small_env(tmp_path: Path):
    return b2b.make_env(
        "OfficeSmall",
        split="train",
        index=0,
        task="task_occ_e0",
        run_period="winter",
        timesteps_per_hour=TIMESTEPS_PER_HOUR,
        max_episode_steps=N_STEPS,
        eplus_output_dir=tmp_path / "eplus",
    )


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
        print(
            f"    Occupancy: min={occ.min():.1f}  max={occ.max():.1f}  "
            f"occupied_steps={n_occupied}  unoccupied_steps={n_unoccupied}"
        )
        print(f"    Target temps: {sorted(unique_targets)}")

        if n_occupied > 0 and n_unoccupied > 0:
            any_zone_varies = True
            assert (
                OCCUPIED_C in unique_targets
            ), f"Zone {zone}: expected {OCCUPIED_C}°C in targets when occupied"
            assert (
                UNOCCUPIED_C in unique_targets
            ), f"Zone {zone}: expected {UNOCCUPIED_C}°C in targets when unoccupied"

    assert any_zone_varies, (
        "Expected at least one zone with both occupied and unoccupied timesteps "
        f"over {N_STEPS} steps (~4 days). Zones: {controlled_zones}"
    )

    env.close()
    print("--- PASS ---")
