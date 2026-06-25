"""Pins the AugmentObservationWithBuildingParams wrapper contract.

Asserts that building parameters are extracted from env metadata, normalized to
[-1, 1], and appended to the observation vector.  Also verifies that missing
metadata raises by default (fail-loud), that ``allow_defaults`` falls back
gracefully, that ``reset()`` re-reads metadata when the underlying env changes,
and that denormalization inverts normalization exactly.  A real-env companion
test confirms the wrapper works against production metadata shapes.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import building2building.api as api_mod
from building2building.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
)
from building2building.types import RewardConfig

_FILLED_REWARD = RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0)


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


class MutableMetadataEnv(gym.Env):
    def __init__(self, low: np.ndarray, high: np.ndarray, metadata: dict):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.metadata = metadata
        self._reset_obs = ((low + high) / 2.0).astype(np.float32)

    def reset(self, **kwargs):  # type: ignore[override]
        return self._reset_obs.copy(), {}

    def step(self, action):  # type: ignore[override]
        return self._reset_obs.copy(), 0.0, True, False, {}


def _full_metadata() -> dict:
    return {
        "area": 150.0,
        "warmup_phases": 3.0,
        "hvac_actuators": ["a1", "a2", "a3"],
        "building_source_metadata": {
            "year_built": 1998.0,
            "geometry_building_num_units": 4.0,
        },
    }


@pytest.mark.quick
def test_augment_building_params_happy_path() -> None:
    env = MutableMetadataEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        metadata=_full_metadata(),
    )
    wrapped = AugmentObservationWithBuildingParams(env)
    assert set(wrapped.building_params.keys()) == {
        "area",
        "warmup_phases",
        "num_actuators",
        "year_built",
        "num_units",
    }
    assert np.all(np.isfinite(wrapped.normalized_params))
    assert np.all(wrapped.normalized_params >= -1.0)
    assert np.all(wrapped.normalized_params <= 1.0)


@pytest.mark.quick
def test_augment_building_params_missing_metadata_raises_by_default() -> None:
    env = MutableMetadataEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        metadata={"warmup_phases": 3.0, "hvac_actuators": ["a1"]},
    )
    with pytest.raises(KeyError, match="area"):
        AugmentObservationWithBuildingParams(env)


@pytest.mark.quick
def test_augment_building_params_allow_defaults_preserves_legacy_behavior(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = MutableMetadataEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        metadata={"warmup_phases": 3.0, "hvac_actuators": ["a1"]},
    )
    wrapped = AugmentObservationWithBuildingParams(env, allow_defaults=True)
    assert wrapped.building_params["area"] == 100.0
    assert "using default" in caplog.text
    assert "area" in caplog.text


@pytest.mark.quick
def test_augment_building_params_reset_reextracts_metadata_and_rebuilds_space() -> None:
    env = MutableMetadataEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        metadata=_full_metadata(),
    )
    wrapped = AugmentObservationWithBuildingParams(env)
    obs1, _ = wrapped.reset()
    assert obs1.shape == (7,)

    env.observation_space = gym.spaces.Box(
        low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        dtype=np.float32,
    )
    env._reset_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    env.metadata = {
        "area": 210.0,
        "warmup_phases": 5.0,
        "hvac_actuators": ["a1"],
        "building_source_metadata": {
            "year_built": 2005.0,
            "geometry_building_num_units": 2.0,
        },
    }

    obs2, _ = wrapped.reset()
    assert obs2.shape == (8,)
    assert wrapped.building_params["area"] == 210.0
    assert wrapped.observation_space.shape == (8,)


@pytest.mark.quick
def test_augment_building_params_real_env(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    """Companion: wrapper must augment the real env's observation_space correctly.

    Tests structural invariants without calling reset() — the fixture env's
    EnergyPlus simulation is only exercised by long tests.
    """
    _patch_registry(monkeypatch, fixture_registry)
    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=_FILLED_REWARD,
        max_episode_steps=4,
    )
    try:
        inner_dim = env.observation_space.shape[0]
        # allow_defaults because hvac_actuators is populated during reset(), not before
        wrapped = AugmentObservationWithBuildingParams(env, allow_defaults=True)
        n_params = len(wrapped.building_params)
        assert n_params > 0
        assert wrapped.observation_space.shape[0] == inner_dim + n_params
        assert np.all(wrapped.normalized_params >= -1.0)
        assert np.all(wrapped.normalized_params <= 1.0)
    finally:
        env.close()


@pytest.mark.quick
def test_augment_building_params_denormalize_round_trip() -> None:
    base_env = MutableMetadataEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        metadata=_full_metadata(),
    )
    normalized_env = NormalizeObservation(base_env)
    wrapped = AugmentObservationWithBuildingParams(normalized_env)

    raw_obs = np.array([20.0, 0.0], dtype=np.float32)
    normalized_obs = normalized_env.observation(raw_obs)
    augmented = wrapped.observation(normalized_obs)
    round_trip = wrapped.denormalize(augmented)
    np.testing.assert_allclose(round_trip, raw_obs, rtol=1e-6, atol=1e-6)


@pytest.mark.quick
def test_augment_building_params_output_scale_matches_observation_space() -> None:
    """Param dims are scaled by 1/PARAM_OUTPUT_SCALE; box bounds must agree.

    Regression guard: changing the divisor without updating the observation
    space (or vice-versa) would silently break learning. Verifies both that
    the normalized values land in [-1/SCALE, 1/SCALE] and that the wrapper's
    observation_space exposes the same bounds.
    """
    env = MutableMetadataEnv(
        low=np.array([10.0, -5.0], dtype=np.float32),
        high=np.array([30.0, 5.0], dtype=np.float32),
        metadata=_full_metadata(),
    )
    wrapped = AugmentObservationWithBuildingParams(env)

    scale = AugmentObservationWithBuildingParams.PARAM_OUTPUT_SCALE
    expected = 1.0 / float(scale)
    assert scale > 0

    # Values stay in the scaled range
    assert np.all(wrapped.normalized_params >= -expected - 1e-6)
    assert np.all(wrapped.normalized_params <= expected + 1e-6)

    # Observation space bounds for the param tail match
    n_params = len(wrapped.normalized_params)
    param_low_tail = wrapped.observation_space.low[-n_params:]
    param_high_tail = wrapped.observation_space.high[-n_params:]
    np.testing.assert_allclose(param_low_tail, -expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(param_high_tail, expected, rtol=0, atol=1e-6)
