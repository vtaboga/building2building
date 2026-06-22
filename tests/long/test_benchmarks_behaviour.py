"""Behavior smoke-tests for benchmark env factories.

These checks stay in the ``long`` tier because benchmark factories use the real
data registry and fan out to real building/task combinations. Runtime cost is
driven by that real-registry fanout, not by rollout length.
"""

from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import pytest

from building2building.benchmarks import (
    ActionSpaceTransfer,
    CrossDomainGeneralization,
    DynamicsAdaptation,
    GoalAdaptation,
)
from building2building.benchmarks.base import BenchmarkProblem

pytestmark = pytest.mark.long


def _reset_and_step_once(env: gym.Env) -> None:
    env.reset()
    action = env.action_space.sample()
    env.step(action)


@pytest.mark.parametrize(
    "benchmark",
    [
        DynamicsAdaptation(difficulty="easy"),
        GoalAdaptation(),
        CrossDomainGeneralization(difficulty="easy"),
        ActionSpaceTransfer(system_type="unitary"),
    ],
)
def test_benchmark_env_factories_make_working_envs(benchmark: object) -> None:
    train_envs: list[gym.Env] = []
    test_envs: list[gym.Env] = []
    try:
        assert isinstance(benchmark, BenchmarkProblem)
        train_envs = benchmark.make_train_envs(n=1)
        test_envs = benchmark.make_test_envs(n=1)
        for env in _iter_envs(train_envs, test_envs):
            _reset_and_step_once(env)
    finally:
        for env in _iter_envs(train_envs, test_envs):
            env.close()


def _iter_envs(*env_groups: list[gym.Env]) -> Iterable[gym.Env]:
    for group in env_groups:
        yield from group
