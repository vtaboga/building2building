"""Pins the PadObservation zone-split and padding contract.

Asserts that zone features are split from non-zone features using the
``observation_names`` metadata, that zone-padding zeros are inserted between
zone slots and the non-zone tail, that non-zone features always appear at the
end of the padded vector, and that the wrapper rebuilds its layout after
``reset()`` when the underlying env changes shape.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from building2building.simulator.wrappers import PadObservation


class MutablePadEnv(gym.Env):
    def __init__(self, obs: np.ndarray, obs_names: list[str]):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.full(obs.shape, -50.0, dtype=np.float32),
            high=np.full(obs.shape, 50.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._obs = obs.astype(np.float32)
        self.metadata = {"observation_names": obs_names}

    def reset(self, **kwargs):  # type: ignore[override]
        return self._obs.copy(), {}

    def step(self, action):  # type: ignore[override]
        return self._obs.copy(), 0.0, True, False, {}


@pytest.mark.quick
def test_pad_observation_uses_metadata_zone_split() -> None:
    names = [
        "Zone Air Temperature Z1",
        "Zone Air Temperature Z2",
        "Outdoor Air Temperature",
        "Outdoor Air Relative Humidity",
        "Current Time of Day",
        "Day of Week",
        "Day of Year",
        "HVAC Electricity Consumption",
        "HVAC Natural Gas Consumption",
    ]
    env = MutablePadEnv(np.arange(9, dtype=np.float32), names)
    wrapped = PadObservation(env, target_size=10)
    assert wrapped._zone_air_temperature_indices(obs_size=9) == [0, 1]


@pytest.mark.quick
def test_pad_observation_zone_split_is_case_and_whitespace_robust() -> None:
    names = [
        "  zOnE aIr TeMpErAtUrE Z1   ",
        "Zone Air Temperature Z2",
        "Outdoor Air Temperature",
        "Outdoor Air Relative Humidity",
        "Current Time of Day",
        "Day of Week",
        "Day of Year",
        "HVAC Electricity Consumption",
        "HVAC Natural Gas Consumption",
    ]
    env = MutablePadEnv(np.arange(9, dtype=np.float32), names)
    wrapped = PadObservation(env, target_size=10)
    assert wrapped._zone_air_temperature_indices(obs_size=9) == [0, 1]


@pytest.mark.quick
def test_pad_observation_places_non_zone_features_at_end() -> None:
    names = [
        "Zone Air Temperature Z1",
        "Zone Air Temperature Z2",
        "Outdoor Air Temperature",
        "Outdoor Air Relative Humidity",
        "Current Time of Day",
        "Day of Week",
        "Day of Year",
        "HVAC Electricity Consumption",
        "HVAC Natural Gas Consumption",
    ]
    obs = np.array([21.0, 22.0, 6.0, 40.0, 12.0, 3.0, 120.0, 11.0, 8.0], dtype=np.float32)
    env = MutablePadEnv(obs, names)
    wrapped = PadObservation(env, target_size=10)

    padded = wrapped.observation(obs)
    np.testing.assert_allclose(padded[:2], np.array([21.0, 22.0], dtype=np.float32))
    np.testing.assert_allclose(padded[2:3], np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(padded[3:], obs[2:])


@pytest.mark.quick
def test_pad_observation_rebuilds_after_reset_shape_and_metadata_change() -> None:
    names_2_zones = [
        "Zone Air Temperature Z1",
        "Zone Air Temperature Z2",
        "Outdoor Air Temperature",
        "Outdoor Air Relative Humidity",
        "Current Time of Day",
        "Day of Week",
        "Day of Year",
        "HVAC Electricity Consumption",
        "HVAC Natural Gas Consumption",
    ]
    env = MutablePadEnv(np.arange(9, dtype=np.float32), names_2_zones)
    wrapped = PadObservation(env, target_size=10)
    obs1, _ = wrapped.reset()
    assert obs1.shape == (10,)

    names_1_zone = [
        "Zone Air Temperature Z1",
        "Outdoor Air Temperature",
        "Outdoor Air Relative Humidity",
        "Current Time of Day",
        "Day of Week",
        "Day of Year",
        "HVAC Electricity Consumption",
        "HVAC Natural Gas Consumption",
    ]
    env._obs = np.array([24.0, 7.0, 35.0, 14.0, 4.0, 122.0, 10.0, 3.0], dtype=np.float32)
    env.observation_space = gym.spaces.Box(
        low=np.full((8,), -50.0, dtype=np.float32),
        high=np.full((8,), 50.0, dtype=np.float32),
        dtype=np.float32,
    )
    env.metadata = {"observation_names": names_1_zone}

    obs2, _ = wrapped.reset()
    assert obs2.shape == (10,)
    np.testing.assert_allclose(obs2[:1], np.array([24.0], dtype=np.float32))
    np.testing.assert_allclose(obs2[1:3], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(obs2[3:], env._obs[1:])
