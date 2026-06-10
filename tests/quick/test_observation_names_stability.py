"""Pins the observation-name layout for each target-temperature mode.

Asserts that the list of observation names returned by the minimal OfficeMedium
(VAV) fixture env exactly matches the committed snapshot for ``constant``,
``occupancy``, and ``random_schedule`` modes.  This catches any change to the
observation vector structure (added/removed/reordered features) before it
silently breaks trained models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import building2building.api as api_mod
from building2building.types import RewardConfig

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "minimal_officemedium"
_FILLED_REWARD = RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0)


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


@pytest.mark.quick
@pytest.mark.parametrize("target_temperature_mode", ["constant", "occupancy", "random_schedule"])
def test_observation_names_snapshot_minimal_officemedium(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
    target_temperature_mode: str,
) -> None:
    _patch_registry(monkeypatch, fixture_registry)
    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=_FILLED_REWARD,
        target_temperature_mode=target_temperature_mode,
        random_schedule_seed=123,
        max_episode_steps=4,
    )
    expected_path = (
        _FIXTURE_DIR / f"expected_observation_names_{target_temperature_mode}.json"
    )
    expected = json.loads(expected_path.read_text())
    try:
        assert env.unwrapped.metadata["observation_names"] == expected
    finally:
        env.close()
