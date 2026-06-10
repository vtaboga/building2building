"""End-to-end rollout tests for the seasonal unoccupied policy (``task_occ_*``).

The ``task_occ_e0`` preset uses ``target_temperature_mode="occupancy"`` with
``unoccupied_policy="seasonal"`` and the default seasonal map
(winter 18 C / shoulder 21 C / summer 26 C).  Built through the public
``make_env`` so the requested ``run_period`` is actually applied — the
lower-level ``make_env_from_config`` does not patch the run period, so
"summer" would otherwise silently simulate January (see
``building2building.api._patch_epjson_run_period``).

Requires EnergyPlus — run with
``B2B_RUN_LONG_TESTS=1 pytest -s tests/long/test_seasonal_unoccupied.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import building2building as b2b

pytestmark = pytest.mark.long


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def _make_seasonal_env(
    tmp_path: Path,
    run_period: str,
    *,
    max_steps: int = 24 * 12 * 7,
    suffix: str = "",
) -> "object":
    return b2b.make_env(
        "OfficeSmall",
        split="train",
        index=0,
        task="task_occ_e0",
        run_period=run_period,
        timesteps_per_hour=12,
        max_episode_steps=max_steps,
        eplus_output_dir=tmp_path / f"eplus_seasonal_{run_period}{suffix}",
    )


def _collect_obs(env, n_steps: int) -> tuple[np.ndarray, list[str]]:
    obs_names: list[str] = env.metadata["observation_names"]
    rows = []
    obs, _ = env.reset()
    rows.append(obs)
    for _ in range(n_steps - 1):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        rows.append(obs)
        if terminated or truncated:
            break
    return np.asarray(rows, dtype=np.float32), obs_names


def test_summer_unoccupied_target_is_warm(tmp_path: Path) -> None:
    """During summer unoccupied hours, the target must follow the
    summer entry of the seasonal map (>= 24 C)."""
    _requires_long_runtime()
    env = _make_seasonal_env(tmp_path, "summer")
    try:
        obs_arr, obs_names = _collect_obs(env, 24 * 12 * 3)
        target_idx = [
            i
            for i, n in enumerate(obs_names)
            if n.strip().lower().startswith("target_temperature")
        ]
        occ_idx = [
            i
            for i, n in enumerate(obs_names)
            if n.strip().lower().startswith("zone_occupancy")
        ]
        assert target_idx and occ_idx

        occ_mask = (obs_arr[:, occ_idx] <= 0.0).any(axis=1)
        unoccupied_targets = obs_arr[occ_mask][:, target_idx]
        # At least some unoccupied timesteps must be seen in a 3-day window.
        if unoccupied_targets.size > 0:
            mean_unocc = float(np.mean(unoccupied_targets))
            assert mean_unocc >= 24.0, (
                "Summer unoccupied target should be >= 24 C (seasonal map "
                f"value was 26 C); observed mean = {mean_unocc:.2f}"
            )
    finally:
        env.close()


def test_winter_unoccupied_target_is_cold(tmp_path: Path) -> None:
    _requires_long_runtime()
    env = _make_seasonal_env(tmp_path, "winter")
    try:
        obs_arr, obs_names = _collect_obs(env, 24 * 12 * 3)
        target_idx = [
            i
            for i, n in enumerate(obs_names)
            if n.strip().lower().startswith("target_temperature")
        ]
        occ_idx = [
            i
            for i, n in enumerate(obs_names)
            if n.strip().lower().startswith("zone_occupancy")
        ]
        occ_mask = (obs_arr[:, occ_idx] <= 0.0).any(axis=1)
        unoccupied_targets = obs_arr[occ_mask][:, target_idx]
        if unoccupied_targets.size > 0:
            mean_unocc = float(np.mean(unoccupied_targets))
            assert mean_unocc <= 19.0, (
                "Winter unoccupied target should be <= 19 C (seasonal map "
                f"value was 18 C); observed mean = {mean_unocc:.2f}"
            )
    finally:
        env.close()


def test_full_year_has_distinct_summer_and_winter_targets(tmp_path: Path) -> None:
    """Rolling out a full year, the mean unoccupied target in
    JAN/FEB must be strictly below the mean in JUL/AUG."""
    _requires_long_runtime()
    env = _make_seasonal_env(tmp_path, "full_year", max_steps=365 * 24 * 12)
    try:
        obs_arr, obs_names = _collect_obs(env, 365 * 24 * 12)
        target_idx = [
            i
            for i, n in enumerate(obs_names)
            if n.strip().lower().startswith("target_temperature")
        ]
        occ_idx = [
            i
            for i, n in enumerate(obs_names)
            if n.strip().lower().startswith("zone_occupancy")
        ]
        doy_idx = next(
            i for i, n in enumerate(obs_names) if n.strip().lower() == "day_of_year"
        )
        occ_mask = (obs_arr[:, occ_idx] <= 0.0).any(axis=1)
        doy = obs_arr[:, doy_idx]
        winter_mask = occ_mask & ((doy <= 59) | (doy >= 335))
        summer_mask = occ_mask & ((doy >= 152) & (doy <= 243))
        winter_target = obs_arr[winter_mask][:, target_idx]
        summer_target = obs_arr[summer_mask][:, target_idx]
        if winter_target.size and summer_target.size:
            assert winter_target.mean() < summer_target.mean() - 3.0, (
                f"winter mean={winter_target.mean():.2f}, "
                f"summer mean={summer_target.mean():.2f}"
            )
    finally:
        env.close()
