"""Pins the ``make_env`` knobs-and-HVAC-matrix contract.

Asserts that every combination of run period, target-temperature mode, and
building type (one fixture per type, spanning the VAV / Unitary / HeatingOnly
archetypes) produces an env whose ``action_space`` has the expected shape and
whose metadata is well formed.  A second test verifies that
``rescale_action=True`` and ``max_episode_steps`` are honoured.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

import building2building.api as api_mod
from building2building.simulator.action_spaces import hvac_action_space
from building2building.types import RewardConfig


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("minimal_building_dir", "run_period", "target_temperature_mode", "random_schedule_seed"),
    [
        ("minimal_officemedium", "full_year", "occupancy", None),
        ("minimal_officemedium", "winter", "constant", None),
        ("minimal_officemedium", "summer", "random_schedule", 123),
        ("minimal_officesmall", "full_year", "occupancy", None),
        ("minimal_restaurantfastfood", "full_year", "occupancy", None),
        ("minimal_retailstandalone", "full_year", "occupancy", None),
        ("minimal_warehouse", "full_year", "occupancy", None),
        ("minimal_singlefamilyhouse", "full_year", "occupancy", None),
    ],
    indirect=["minimal_building_dir"],
)
def test_make_env_knobs_and_hvac_matrix(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
    run_period: str,
    target_temperature_mode: str,
    random_schedule_seed: int | None,
) -> None:
    _patch_registry(monkeypatch, fixture_registry)

    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0),
        run_period=run_period,
        target_temperature_mode=target_temperature_mode,
        random_schedule_seed=random_schedule_seed,
        max_episode_steps=20,
        rescale_action=False,
    )
    try:
        assert env.observation_space.shape is not None
        assert env.observation_space.shape[0] > 0
        assert env.action_space.shape is not None
        assert env.action_space.shape[0] > 0

        metadata = env.unwrapped.metadata
        assert metadata["controlled_zones"]
        assert metadata["task_config"].run_period.name == run_period
        assert metadata["task_config"].target_temperature_mode == target_temperature_mode
        controlled_zones = set(metadata["controlled_zones"])
        equipment = metadata["hvac_equipment"]
        controlled_actuators = list(
            itertools.chain.from_iterable(
                eq.actuator_descriptions()
                for eq in equipment
                if controlled_zones.intersection(eq.zones())
            )
        )
        n_agent_actuators = len(hvac_action_space(controlled_actuators).agent_actuators)
        assert env.action_space.shape[0] == n_agent_actuators

        assert env._max_episode_steps == 20  # TimeLimit wrapper contract
    finally:
        env.close()


@pytest.mark.quick
def test_make_env_rescale_action_and_max_steps(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    _patch_registry(monkeypatch, fixture_registry)
    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0),
        rescale_action=True,
        max_episode_steps=12,
    )
    try:
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        assert env._max_episode_steps == 12
    finally:
        env.close()
