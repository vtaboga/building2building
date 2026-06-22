"""Pins the PadObservation zone-alignment invariant on real environments.

Serves as the companion test for ``PadObservation`` (T24 wrapper audit): asserts
that after padding, the non-zone tail features appear at the same fixed indices
regardless of the number of zones in the building.  Uses a real env built from
``fixture_registry`` so the metadata and observation-space shape match
production.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import building2building.api as api_mod
from building2building.simulator.wrappers import PadObservation
from building2building.types import RewardConfig

_TARGET_SIZE = 80
_NON_ZONE_FEATURES = 7
_FILLED_REWARD = RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0)
_TAIL_NON_ZONE_NAMES = [
    "time_of_day",
    "day_of_week",
    "day_of_year",
    "outdoor_temperature",
    "outdoor_humidity",
    "energy_gas",
    "energy_electricity",
]


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    "minimal_building_dir",
    ["minimal_officesmall", "minimal_officemedium"],
    indirect=True,
)
def test_pad_observation_keeps_non_zone_tail_at_stable_indices(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    _patch_registry(monkeypatch, fixture_registry)

    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=_FILLED_REWARD,
        max_episode_steps=4,
    )
    try:
        obs_names = env.unwrapped.metadata["observation_names"]
        raw_obs = np.arange(1, len(obs_names) + 1, dtype=env.observation_space.dtype)
        zone_indices = [
            i
            for i, name in enumerate(obs_names)
            if str(name).strip().lower().startswith("zone air temperature")
        ]
        assert obs_names[-_NON_ZONE_FEATURES:] == _TAIL_NON_ZONE_NAMES
        tail_indices = list(range(len(obs_names) - _NON_ZONE_FEATURES, len(obs_names)))

        wrapped = PadObservation(env, target_size=_TARGET_SIZE)
        padded = wrapped.observation(raw_obs)

        np.testing.assert_allclose(
            padded[-_NON_ZONE_FEATURES:],
            raw_obs[tail_indices],
            atol=1e-6,
        )
        assert np.any(np.abs(padded[-_NON_ZONE_FEATURES:]) > 1e-6)

        max_zones = _TARGET_SIZE - _NON_ZONE_FEATURES
        current_num_zones = len(zone_indices)
        np.testing.assert_array_equal(
            padded[current_num_zones:max_zones],
            np.zeros(max_zones - current_num_zones, dtype=padded.dtype),
        )
    finally:
        env.close()
