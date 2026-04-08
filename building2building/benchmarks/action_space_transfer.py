"""Action-space transfer benchmark (Section 4, Table: Action-space transfer).

Building dynamics and reward stay fixed; controllable actuators change
between training and test.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from building2building.benchmarks.base import BenchmarkProblem


class ActionSpaceTransfer(BenchmarkProblem):
    """Action-space transfer benchmark.

    The agent trains on a building with one set of controllable
    actuators and is tested on the *same building* with a different
    (expanded or reduced) actuator set.

    Args:
        system_type: HVAC system type (``"unitary"`` or ``"central"``).
        direction: Whether the test set has more (``"expand"``) or
            fewer (``"reduce"``) actuators than training.
        task: Named task preset.
        building_type: Building type to use.
        split_index: Index within the split.
    """

    def __init__(
        self,
        system_type: Literal["unitary", "central"] = "unitary",
        direction: Literal["expand", "reduce"] = "expand",
        task: str = "task1",
        building_type: str = "OfficeSmall",
        split_index: int = 0,
    ) -> None:
        self.system_type = system_type
        self.direction = direction
        self.task = task
        self.building_type = building_type
        self.split_index = split_index

    def make_train_env(self, **kwargs: object) -> gym.Env:
        """Create a single training environment."""
        from building2building.api import new_make_env

        return new_make_env(
            building_type=self.building_type,  # type: ignore[arg-type]
            split="train",
            index=self.split_index,
            task=self.task,
            **kwargs,  # type: ignore[arg-type]
        )

    def make_test_env(self, **kwargs: object) -> gym.Env:
        """Create a single test environment."""
        from building2building.api import new_make_env

        return new_make_env(
            building_type=self.building_type,  # type: ignore[arg-type]
            split="test",
            index=self.split_index,
            task=self.task,
            **kwargs,  # type: ignore[arg-type]
        )

    def make_train_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create training environments (one by default)."""
        return [self.make_train_env() for _ in range(n or 1)]

    def make_test_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create test environments (one by default)."""
        return [self.make_test_env() for _ in range(n or 1)]
