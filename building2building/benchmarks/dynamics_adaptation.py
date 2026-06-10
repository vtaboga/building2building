"""Dynamics adaptation benchmark (Section 4, Table: Dynamics adaptation).

Reward and action space stay fixed; building dynamics vary between
training and test sets.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from building2building.benchmarks.base import BenchmarkProblem
from building2building.data.download import BuildingType

DYNAMICS_ADAPTATION_PRESETS: dict[str, dict[str, object]] = {
    "easy": {"building_type": "SingleFamilyHouse", "action_dim": 2},
    "medium": {"building_type": "OfficeSmall", "action_dim": 10},
    "hard": {"building_type": "OfficeMedium", "action_dim": 36},
}


class DynamicsAdaptation(BenchmarkProblem):
    """Dynamics adaptation benchmark.

    Agents are trained and tested on *different buildings* of the same
    type.  The task (reward, action space) stays fixed; only the
    building dynamics change.

    Args:
        difficulty: One of ``"easy"``, ``"medium"``, ``"hard"``.
            Determines the building type and action dimensionality.
        task: Named task preset.
        n_train: Number of training buildings.
        n_test: Number of test buildings.
        seed: Random seed for split selection.
    """

    def __init__(
        self,
        difficulty: Literal["easy", "medium", "hard"] = "easy",
        task: str = "task_const_e0",
        n_train: int = 900,
        n_test: int = 71,
        seed: int = 0,
    ) -> None:
        if difficulty not in DYNAMICS_ADAPTATION_PRESETS:
            raise ValueError(
                f"Unknown difficulty {difficulty!r}. "
                f"Choose from {sorted(DYNAMICS_ADAPTATION_PRESETS.keys())}"
            )
        preset = DYNAMICS_ADAPTATION_PRESETS[difficulty]
        self.building_type: BuildingType = preset["building_type"]  # type: ignore[assignment]
        self.task = task
        self.n_train = n_train
        self.n_test = n_test
        self.seed = seed
        self.difficulty = difficulty

    def train_building_ids(self) -> list[str]:
        """Return the list of training building IDs."""
        from building2building.data.registry import get_registry

        registry = get_registry()
        ids = registry.list_buildings(self.building_type, "train")
        return ids[: self.n_train]

    def test_building_ids(self) -> list[str]:
        """Return the list of test building IDs."""
        from building2building.data.registry import get_registry

        registry = get_registry()
        ids = registry.list_buildings(self.building_type, "test")
        return ids[: self.n_test]

    def make_train_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create training environments, one per training building."""
        from building2building.api import make_env

        ids = self.train_building_ids()
        if n is not None:
            ids = ids[:n]
        return [
            make_env(
                building_type=self.building_type,
                building_id=bid,
                task=self.task,
            )
            for bid in ids
        ]

    def make_test_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create test environments, one per test building."""
        from building2building.api import make_env

        ids = self.test_building_ids()
        if n is not None:
            ids = ids[:n]
        return [
            make_env(
                building_type=self.building_type,
                building_id=bid,
                task=self.task,
            )
            for bid in ids
        ]
