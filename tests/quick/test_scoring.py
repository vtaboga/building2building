"""Tests for building2building.scoring — normalized score computation."""

from __future__ import annotations

import pytest

from building2building import scoring


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Ensure each test starts with a clean cache."""
    scoring._cache = None


@pytest.mark.quick
class TestComputeNormalizedScore:
    @pytest.fixture(autouse=True)
    def _inject_baselines(self) -> None:
        """Inject a controlled baseline cache for deterministic tests."""
        scoring._cache = {
            ("OfficeSmall", "task1", "full_year", "OfficeSmall-0001"): -30000.0,
            ("OfficeSmall", "task1", "full_year", "OfficeSmall-0002"): -25000.0,
            ("OfficeSmall", "task1", "winter", "OfficeSmall-0001"): -12000.0,
            ("OfficeSmall", "task1", "summer", "OfficeSmall-0001"): -18000.0,
            ("Warehouse", "task1", "full_year", "Warehouse-0001"): -40000.0,
            (
                "SingleFamilyHouse",
                "task1",
                "full_year",
                "SingleFamilyHouse-0001",
            ): -10000.0,
        }

    def test_by_building_id(self) -> None:
        score = scoring.compute_normalized_score(
            cumulative_return=-30000.0,
            building_type="OfficeSmall",
            task="task1",
            run_period="full_year",
            building_id="OfficeSmall-0001",
        )
        assert score == pytest.approx(1.0)

    def test_missing_building_id_argument_raises(self) -> None:
        with pytest.raises(TypeError, match="building_id"):
            scoring.compute_normalized_score(
                cumulative_return=-30000.0,
                building_type="OfficeSmall",
                task="task1",
                run_period="full_year",
            )  # type: ignore[call-arg]

    def test_none_building_id_raises(self) -> None:
        with pytest.raises(ValueError, match="building_id must be specified"):
            scoring.compute_normalized_score(
                cumulative_return=-30000.0,
                building_type="OfficeSmall",
                task="task1",
                run_period="full_year",
                building_id=None,  # type: ignore[arg-type]
            )

    def test_better_than_baseline(self) -> None:
        score = scoring.compute_normalized_score(
            cumulative_return=-15000.0,
            building_type="OfficeSmall",
            task="task1",
            run_period="full_year",
            building_id="OfficeSmall-0001",
        )
        assert score == pytest.approx(-15000.0 / -30000.0)

    def test_winter_vs_summer_differ(self) -> None:
        winter = scoring.compute_normalized_score(
            cumulative_return=-12000.0,
            building_type="OfficeSmall",
            task="task1",
            run_period="winter",
            building_id="OfficeSmall-0001",
        )
        summer = scoring.compute_normalized_score(
            cumulative_return=-12000.0,
            building_type="OfficeSmall",
            task="task1",
            run_period="summer",
            building_id="OfficeSmall-0001",
        )
        assert winter == pytest.approx(1.0)
        assert summer == pytest.approx(-12000.0 / -18000.0)
        assert winter != summer

    def test_missing_building_id_raises(self) -> None:
        with pytest.raises(KeyError, match="No baseline return found"):
            scoring.compute_normalized_score(
                cumulative_return=-1.0,
                building_type="OfficeSmall",
                task="task1",
                run_period="full_year",
                building_id="OfficeSmall-9999",
            )

    def test_missing_run_period_raises(self) -> None:
        with pytest.raises(KeyError, match="No baseline return found"):
            scoring.compute_normalized_score(
                cumulative_return=-1.0,
                building_type="Warehouse",
                task="task1",
                run_period="winter",
                building_id="Warehouse-0001",
            )

    def test_zero_baseline_returns_raw(self) -> None:
        scoring._cache = {("ZeroType", "task1", "full_year", "ZT-001"): 0.0}
        score = scoring.compute_normalized_score(
            cumulative_return=-500.0,
            building_type="ZeroType",  # type: ignore[arg-type]
            task="task1",
            run_period="full_year",
            building_id="ZT-001",
        )
        assert score == -500.0
