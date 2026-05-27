from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from building2building.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
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
