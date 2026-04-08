"""Tests for building2building.scoring — normalized score computation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from building2building import scoring


@pytest.fixture(autouse=True)
def _clear_baseline_cache() -> None:
    """Ensure each test starts with a clean cache."""
    scoring._baseline_cache = None
    scoring._baseline_cache_no_task = None


@pytest.mark.quick
class TestComputeNormalizedScore:
    @pytest.fixture(autouse=True)
    def _inject_baselines(self) -> None:
        """Inject a controlled baseline cache for deterministic tests."""
        scoring._baseline_cache = {
            ("OfficeSmall", "task1", "OfficeSmall-0001"): -30000.0,
            ("OfficeSmall", "task1", "OfficeSmall-0002"): -25000.0,
            ("Warehouse", "task1", "Warehouse-0001"): -40000.0,
            ("SingleFamilyHouse", "task1", "SingleFamilyHouse-0001"): -10000.0,
        }
        scoring._baseline_cache_no_task = {
            ("OfficeSmall", "OfficeSmall-0001"): -30000.0,
            ("OfficeSmall", "OfficeSmall-0002"): -25000.0,
            ("Warehouse", "Warehouse-0001"): -40000.0,
            ("SingleFamilyHouse", "SingleFamilyHouse-0001"): -10000.0,
        }

    def test_by_building_id(self) -> None:
        score = scoring.compute_normalized_score(
            cumulative_return=-30000.0,
            building_type="OfficeSmall",
            task="task1",
            building_id="OfficeSmall-0001",
        )
        assert score == pytest.approx(1.0)

    def test_by_building_type_average(self) -> None:
        avg = (-30000.0 + -25000.0) / 2
        score = scoring.compute_normalized_score(
            cumulative_return=avg,
            building_type="OfficeSmall",
            task="task1",
        )
        assert score == pytest.approx(1.0)

    def test_better_than_baseline(self) -> None:
        score = scoring.compute_normalized_score(
            cumulative_return=-15000.0,
            building_type="OfficeSmall",
            task="task1",
            building_id="OfficeSmall-0001",
        )
        assert score == pytest.approx(-15000.0 / -30000.0)

    def test_missing_building_id_raises(self) -> None:
        with pytest.raises(KeyError, match="No baseline return found"):
            scoring.compute_normalized_score(
                cumulative_return=-1.0,
                building_type="OfficeSmall",
                task="task1",
                building_id="OfficeSmall-9999",
            )

    def test_missing_building_type_raises(self) -> None:
        with pytest.raises(KeyError, match="No baseline returns found"):
            scoring.compute_normalized_score(
                cumulative_return=-1.0,
                building_type="NonExistent",  # type: ignore[arg-type]
                task="task1",
            )

    def test_zero_baseline_returns_raw(self) -> None:
        scoring._baseline_cache = {("ZeroType", "task1", "ZT-001"): 0.0}
        scoring._baseline_cache_no_task = {("ZeroType", "ZT-001"): 0.0}
        score = scoring.compute_normalized_score(
            cumulative_return=-500.0,
            building_type="ZeroType",  # type: ignore[arg-type]
            task="task1",
            building_id="ZT-001",
        )
        assert score == -500.0
