"""Cross-domain generalization benchmark (Section 4, Table: Cross-domain).

Agents are trained on one building type and tested on a *different*
building type.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from building2building.benchmarks.base import BenchmarkProblem
from building2building.data.download import BuildingType

CROSS_DOMAIN_PRESETS: dict[str, dict[str, str]] = {
    "easy": {"train_type": "RetailStandalone", "test_type": "OfficeSmall"},
    "medium": {"train_type": "RetailStandalone", "test_type": "Warehouse"},
    "hard": {"train_type": "OfficeSmall", "test_type": "OfficeMedium"},
}


class CrossDomainGeneralization(BenchmarkProblem):
    """Cross-domain generalization benchmark.

    Agents are trained on buildings of one type and tested on buildings
    of a *different* type.

    Args:
        difficulty: One of ``"easy"``, ``"medium"``, ``"hard"``.
            Determines the train/test building type pair.
        task: Named task preset.
        n_train: Number of training buildings.
        n_test: Number of test buildings.
        seed: Random seed for building selection.
    """

    def __init__(
        self,
        difficulty: Literal["easy", "medium", "hard"] = "easy",
        task: str = "task_const_e0",
        n_train: int = 8,
        n_test: int = 8,
        seed: int = 0,
    ) -> None:
        if difficulty not in CROSS_DOMAIN_PRESETS:
            raise ValueError(
                f"Unknown difficulty {difficulty!r}. "
                f"Choose from {sorted(CROSS_DOMAIN_PRESETS.keys())}"
            )
        preset = CROSS_DOMAIN_PRESETS[difficulty]
        self.train_type: BuildingType = preset["train_type"]  # type: ignore[assignment]
        self.test_type: BuildingType = preset["test_type"]  # type: ignore[assignment]
        self.task = task
        self.n_train = n_train
        self.n_test = n_test
        self.seed = seed
        self.difficulty = difficulty

    def make_train_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create training environments from the training building type."""
        from building2building.api import make_env

        count = n if n is not None else self.n_train
        return [
            make_env(
                building_type=self.train_type,
                split="train",
                index=i,
                task=self.task,
            )
            for i in range(count)
        ]

    def make_test_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create test environments from the test building type."""
        from building2building.api import make_env

        count = n if n is not None else self.n_test
        return [
            make_env(
                building_type=self.test_type,
                split="test",
                index=i,
                task=self.task,
            )
            for i in range(count)
        ]
