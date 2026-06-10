"""Pins the ResampleBuildingWrapper contract.

Asserts that the wrapper raises on empty index lists, that a single-index list
yields a stable env on every reset, that a multi-index list swaps the
underlying env on reset, that episode counters reset correctly across swaps, and
that index-out-of-bounds errors are warned about and deferred to the next reset
rather than crashing immediately.
"""

from __future__ import annotations

import random
import sys

import gymnasium as gym
import numpy as np
import pytest

from building2building.simulator.wrappers import ResampleBuildingOnResetWrapper


class FactoryMockEnv(gym.Env):
    def __init__(self, index: int, raise_index_error: bool = False):
        super().__init__()
        self.index = index
        self.raise_index_error = raise_index_error
        self.close_called = False
        self.step_count = 0
        self.metadata = {"building_source_metadata": {"building_id": f"id-{index}"}}
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):  # type: ignore[override]
        self.step_count = 0
        return np.array([float(self.index), 0.0], dtype=np.float32), {}

    def step(self, action):  # type: ignore[override]
        self.step_count += 1
        if self.raise_index_error:
            raise IndexError("actuator index mismatch")
        obs = np.array([float(self.index), float(self.step_count)], dtype=np.float32)
        return obs, 1.0, False, False, {}

    def close(self) -> None:
        self.close_called = True
        super().close()


@pytest.mark.quick
def test_resample_wrapper_raises_on_empty_indices() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ResampleBuildingOnResetWrapper(lambda idx: FactoryMockEnv(idx), [])


@pytest.mark.quick
def test_resample_wrapper_single_index_is_stable() -> None:
    calls: list[int] = []

    def factory(idx: int) -> gym.Env:
        calls.append(idx)
        return FactoryMockEnv(idx)

    wrapped = ResampleBuildingOnResetWrapper(factory, [0])
    for _ in range(5):
        wrapped.reset()
    assert calls == [0]


@pytest.mark.quick
def test_resample_wrapper_multi_index_swaps_and_closes_old_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = iter([0, 1])
    monkeypatch.setattr(random, "choice", lambda _: next(sequence))

    calls: list[int] = []
    created: list[FactoryMockEnv] = []

    def factory(idx: int) -> gym.Env:
        calls.append(idx)
        env = FactoryMockEnv(idx)
        created.append(env)
        return env

    wrapped = ResampleBuildingOnResetWrapper(factory, [0, 1, 2])
    wrapped.reset()

    assert calls == [0, 1]
    assert created[0].close_called is True


@pytest.mark.quick
def test_resample_wrapper_episode_counters_reset() -> None:
    wrapped = ResampleBuildingOnResetWrapper(lambda idx: FactoryMockEnv(idx), [0])
    wrapped.reset()
    for _ in range(5):
        wrapped.step(np.array([0.0], dtype=np.float32))

    wrapped.reset()
    assert wrapped._episode_reward == 0.0
    assert wrapped._episode_steps == 0
    assert wrapped._episode_count == 2


@pytest.mark.quick
def test_resample_wrapper_index_error_warns_and_defers_resample_to_next_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = iter([0, 0, 1])
    monkeypatch.setattr(random, "choice", lambda _: next(sequence))

    calls: list[int] = []

    def factory(idx: int) -> gym.Env:
        calls.append(idx)
        return FactoryMockEnv(idx, raise_index_error=(idx == 0))

    wrapped = ResampleBuildingOnResetWrapper(factory, [0, 1])
    wrapped.reset()

    with pytest.warns(RuntimeWarning, match="actuator"):
        obs, reward, terminated, truncated, _ = wrapped.step(
            np.array([0.0], dtype=np.float32)
        )

    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert obs.shape == (2,)

    wrapped.reset()
    assert calls == [0, 1]


@pytest.mark.quick
def test_resample_wrapper_wandb_noop_when_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    wrapped = ResampleBuildingOnResetWrapper(lambda idx: FactoryMockEnv(idx), [0])
    wrapped.reset()
    wrapped.step(np.array([0.0], dtype=np.float32))
