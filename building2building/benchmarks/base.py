"""Abstract base class for benchmark problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import gymnasium as gym

EnvConstructor = Callable[[], gym.Env]


class BenchmarkProblem(ABC):
    """Base class for named benchmark problems from the B2B paper.

    Each benchmark exposes methods to create train and test environments
    that reproduce a specific experimental setting from the paper.
    """

    @abstractmethod
    def make_train_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create training environments."""
        ...

    @abstractmethod
    def make_test_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create test environments."""
        ...
