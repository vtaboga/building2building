"""Tests for building2building.benchmarks — all 4 benchmark classes."""

from __future__ import annotations

import pytest

from building2building.benchmarks import (
    ActionSpaceTransfer,
    CrossDomainGeneralization,
    DynamicsAdaptation,
    GoalAdaptation,
)
from building2building.benchmarks.cross_domain import CROSS_DOMAIN_PRESETS
from building2building.benchmarks.dynamics_adaptation import DYNAMICS_ADAPTATION_PRESETS


@pytest.mark.quick
class TestGoalAdaptation:
    def test_defaults(self) -> None:
        bm = GoalAdaptation()
        assert bm.building_type == "OfficeSmall"
        assert bm.train_task == "task1"
        assert bm.test_task == "task2"
        assert bm.run_period == "full_year"

    def test_custom_params(self) -> None:
        bm = GoalAdaptation(
            building_type="Warehouse",
            train_task="task3",
            test_task="task4",
            run_period="winter",
        )
        assert bm.building_type == "Warehouse"
        assert bm.train_task == "task3"
        assert bm.test_task == "task4"


@pytest.mark.quick
class TestDynamicsAdaptation:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_valid_difficulties(self, difficulty: str) -> None:
        bm = DynamicsAdaptation(difficulty=difficulty)  # type: ignore[arg-type]
        preset = DYNAMICS_ADAPTATION_PRESETS[difficulty]
        assert bm.building_type == preset["building_type"]

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown difficulty"):
            DynamicsAdaptation(difficulty="extreme")  # type: ignore[arg-type]

    def test_easy_is_single_family_house(self) -> None:
        bm = DynamicsAdaptation(difficulty="easy")
        assert bm.building_type == "SingleFamilyHouse"

    def test_hard_is_office_medium(self) -> None:
        bm = DynamicsAdaptation(difficulty="hard")
        assert bm.building_type == "OfficeMedium"

    def test_custom_n_train_test(self) -> None:
        bm = DynamicsAdaptation(n_train=10, n_test=5)
        assert bm.n_train == 10
        assert bm.n_test == 5


@pytest.mark.quick
class TestActionSpaceTransfer:
    def test_defaults(self) -> None:
        bm = ActionSpaceTransfer()
        assert bm.system_type == "unitary"
        assert bm.direction == "expand"
        assert bm.task == "task1"

    def test_custom_system_type(self) -> None:
        bm = ActionSpaceTransfer(system_type="central", direction="reduce")
        assert bm.system_type == "central"
        assert bm.direction == "reduce"


@pytest.mark.quick
class TestCrossDomainGeneralization:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_valid_difficulties(self, difficulty: str) -> None:
        bm = CrossDomainGeneralization(difficulty=difficulty)  # type: ignore[arg-type]
        preset = CROSS_DOMAIN_PRESETS[difficulty]
        assert bm.train_type == preset["train_type"]
        assert bm.test_type == preset["test_type"]

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown difficulty"):
            CrossDomainGeneralization(difficulty="ultra")  # type: ignore[arg-type]

    def test_easy_preset(self) -> None:
        bm = CrossDomainGeneralization(difficulty="easy")
        assert bm.train_type == "RetailStandalone"
        assert bm.test_type == "OfficeSmall"

    def test_custom_n_train_test(self) -> None:
        bm = CrossDomainGeneralization(n_train=2, n_test=3)
        assert bm.n_train == 2
        assert bm.n_test == 3
