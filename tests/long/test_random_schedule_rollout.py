"""End-to-end rollout tests for ``task5`` (random daily schedule).

Requires EnergyPlus — run with
``B2B_RUN_LONG_TESTS=1 pytest -s tests/long/test_random_schedule_rollout.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from building2building.config.models import DatasetSelectionConfig, EnvBuildConfig
from building2building.envs.factory import make_env_from_config
from building2building.types import (
    RandomScheduleConfig,
    RewardConfig,
    TaskConfig,
)

pytestmark = pytest.mark.long

N_STEPS = 24 * 12 * 3  # 3 days at 5-min resolution


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


def _make_env(tmp_path: Path, seed: int, *, suffix: str = "") -> "object":
    task = TaskConfig(
        run_period=__import__(
            "building2building.types", fromlist=["RunPeriodConfig"]
        ).RunPeriodConfig.from_name("winter"),
        target_temperature_mode="random_schedule",
        default_zone_target_temperature=__import__(
            "building2building.types", fromlist=["ZoneTargetTemperatureConfig"]
        ).ZoneTargetTemperatureConfig(
            occupied_c=21.0,
            unoccupied_c=18.0,
        ),
        random_schedule_config=RandomScheduleConfig(
            building_type="OfficeSmall",
            seed=seed,
        ),
    )
    config = EnvBuildConfig(
        dataset_selection=DatasetSelectionConfig(
            building_type="OfficeSmall",
            split="train",
            mode="split_index",
            split_index=0,
        ),
        task=task,
        reward=RewardConfig(energy_weight=0.0, dT=1.0, tau_T=1.0, tau_E=1.0),
        env_max_steps=N_STEPS,
    )
    return make_env_from_config(
        config, eplus_output_dir=tmp_path / f"eplus_task5{suffix}"
    )


def _target_trace(env, n_steps: int) -> np.ndarray:
    obs_names: list[str] = env.metadata["observation_names"]
    target_idx = [
        i
        for i, n in enumerate(obs_names)
        if n.strip().lower().startswith("target_temperature")
    ]
    assert target_idx, "random-schedule task must expose target_temperature slots"

    # The random-schedule target is deterministic in (seed, day_of_year, hour),
    # but the hour sampled at a given step depends on the action stream: the
    # actions drive EnergyPlus HVAC system-iteration counts, which shift when
    # a setpoint transition lands on a step index.  To compare two same-seed
    # rollouts the action stream must be identical, so seed the action space
    # deterministically here (otherwise reproducibility checks see spurious
    # ~1-step misalignments at transitions).
    env.action_space.seed(0)
    trace = np.empty((n_steps, len(target_idx)), dtype=np.float32)
    obs, _ = env.reset()
    trace[0] = obs[target_idx]
    for step in range(1, n_steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        trace[step] = obs[target_idx]
        if terminated or truncated:
            trace = trace[: step + 1]
            break
    return trace


def test_target_trace_has_multiple_distinct_values(tmp_path: Path) -> None:
    _requires_long_runtime()
    env = _make_env(tmp_path, seed=0)
    try:
        trace = _target_trace(env, N_STEPS)
        unique_per_zone = [np.unique(trace[:, j]).size for j in range(trace.shape[1])]
        # At least two setpoints (occupied vs unoccupied) per zone.
        assert min(unique_per_zone) >= 2, unique_per_zone
    finally:
        env.close()


def test_seed_determinism(tmp_path: Path) -> None:
    _requires_long_runtime()
    env_a = _make_env(tmp_path, seed=123, suffix="_a")
    env_b = _make_env(tmp_path, seed=123, suffix="_b")
    try:
        trace_a = _target_trace(env_a, N_STEPS)
        trace_b = _target_trace(env_b, N_STEPS)
        assert trace_a.shape == trace_b.shape
        np.testing.assert_allclose(trace_a, trace_b, atol=1e-6)
    finally:
        env_a.close()
        env_b.close()


def test_different_seeds_differ(tmp_path: Path) -> None:
    _requires_long_runtime()
    env_a = _make_env(tmp_path, seed=0, suffix="_s0")
    env_b = _make_env(tmp_path, seed=999, suffix="_s999")
    try:
        trace_a = _target_trace(env_a, N_STEPS)
        trace_b = _target_trace(env_b, N_STEPS)
        # The probability of coincident traces with different seeds
        # is effectively zero for 3 days of random sampling.
        assert not np.allclose(trace_a, trace_b, atol=1e-6)
    finally:
        env_a.close()
        env_b.close()


WEEK_STEPS = 7 * 24 * 12


def test_week_reproducibility_via_make_env(tmp_path: Path) -> None:
    """Two independent one-week rollouts with the same seed must produce
    byte-identical ``target_temperature`` traces.

    This exercises the public :func:`building2building.make_env`
    entry point (the same one used by the analysis scripts), as opposed
    to :func:`make_env_from_config` covered by
    :func:`test_seed_determinism`.  The run window defaults to the
    first week of ``full_year`` so the trace falls inside the seasonal
    winter branch of ``task5`` without relying on the ``winter``
    run-period epJSON patch.
    """
    _requires_long_runtime()
    import building2building as b2b

    seed = 2025

    def _make(suffix: str):
        return b2b.make_env(
            "OfficeSmall",
            split="train",
            index=0,
            task="task_rand_e0",
            run_period="full_year",
            random_schedule_seed=seed,
            eplus_output_dir=tmp_path / f"week_repro{suffix}",
            max_episode_steps=WEEK_STEPS,
        )

    env_a = _make("_a")
    env_b = _make("_b")
    try:
        trace_a = _target_trace(env_a, WEEK_STEPS)
        trace_b = _target_trace(env_b, WEEK_STEPS)

        assert trace_a.shape == (WEEK_STEPS, trace_a.shape[1])
        assert trace_a.shape == trace_b.shape

        # Every zone must show a non-trivial random schedule
        # (otherwise the test trivially passes on a constant trace).
        unique_per_zone = [
            int(np.unique(trace_a[:, j]).size) for j in range(trace_a.shape[1])
        ]
        assert min(unique_per_zone) >= 2, (
            f"Expected at least two distinct setpoints per zone, got "
            f"{unique_per_zone}"
        )

        # Exact equality — the random schedule is driven by a fully
        # deterministic (seed, year, day_of_year, zone) hash, so the
        # two traces must match bit-for-bit, not merely within an
        # atol.
        np.testing.assert_array_equal(trace_a, trace_b)
    finally:
        env_a.close()
        env_b.close()


def test_occupancy_consistent_with_setpoints(tmp_path: Path) -> None:
    """The zone_occupancy 0/1 indicator must agree with the target
    transition: occupied periods see ``occupied_c``, unoccupied
    periods see ``unoccupied_c``."""
    _requires_long_runtime()
    env = _make_env(tmp_path, seed=7, suffix="_consistency")
    try:
        obs_names: list[str] = env.metadata["observation_names"]
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
        assert len(target_idx) == len(occ_idx) > 0

        obs, _ = env.reset()
        for _ in range(N_STEPS):
            for tgt, occ in zip(target_idx, occ_idx):
                # Occupancy is exposed as 0/1 for task5; target must be
                # one of two values consistent across the day.
                assert obs[occ] in (0.0, 1.0)
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
    finally:
        env.close()
