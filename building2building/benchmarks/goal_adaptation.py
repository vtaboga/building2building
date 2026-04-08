"""Goal adaptation benchmark (Section 4, Table: Goal adaptation).

Training and test environments use the same building and control
interface, but the reward function / task changes between them.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from building2building.benchmarks.base import BenchmarkProblem
from building2building.data.download import BuildingType


class GoalAdaptation(BenchmarkProblem):
    """Goal adaptation benchmark.

    The agent trains with one task (reward / temperature target) and is
    tested with a different task on the *same* building.

    Args:
        building_type: Building type to use.
        split_index: Index within the train split.
        train_task: Named task preset for training.
        test_task: Named task preset for testing.
        run_period: Simulation run period.
    """

    def __init__(
        self,
        building_type: BuildingType = "OfficeSmall",
        split_index: int = 0,
        train_task: str = "task1",
        test_task: str = "task2",
        run_period: str = "full_year",
    ) -> None:
        self.building_type = building_type
        self.split_index = split_index
        self.train_task = train_task
        self.test_task = test_task
        self.run_period = run_period

    def make_train_env(self, **kwargs: object) -> gym.Env:
        """Create a single training environment."""
        from building2building.api import new_make_env

        return new_make_env(
            building_type=self.building_type,
            split="train",
            index=self.split_index,
            task=self.train_task,
            run_period=self.run_period,
            **kwargs,  # type: ignore[arg-type]
        )

    def make_test_env(self, **kwargs: object) -> gym.Env:
        """Create a single test environment."""
        from building2building.api import new_make_env

        return new_make_env(
            building_type=self.building_type,
            split="train",
            index=self.split_index,
            task=self.test_task,
            run_period=self.run_period,
            **kwargs,  # type: ignore[arg-type]
        )

    def make_train_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create training environments (one by default)."""
        return [self.make_train_env() for _ in range(n or 1)]

    def make_test_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create test environments (one by default)."""
        return [self.make_test_env() for _ in range(n or 1)]
