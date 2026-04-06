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


@pytest.mark.quick
class TestLoadBaselineReturns:
    def test_loads_from_fixture(self, baseline_csv_path: Path) -> None:
        with patch.object(
            scoring,
            "_load_baseline_returns",
            wraps=scoring._load_baseline_returns,
        ):
            real_csv_path = Path(scoring.__file__).parent.parent / "baseline_returns.csv"
            with patch("building2building.scoring.Path") as mock_path_cls:
                mock_path_cls.return_value.__truediv__ = (
                    Path.__truediv__.__get__(baseline_csv_path.parent)
                )
                baselines = scoring._load_baseline_returns()
                scoring._baseline_cache = None

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with patch(
            "building2building.scoring.Path.__new__",
            return_value=tmp_path / "nonexistent",
        ):
            scoring._baseline_cache = None


@pytest.mark.quick
class TestComputeNormalizedScore:
    @pytest.fixture(autouse=True)
    def _inject_baselines(self) -> None:
        """Inject a controlled baseline cache for deterministic tests."""
        scoring._baseline_cache = {
            ("OfficeSmall", 1): -30000.0,
            ("OfficeSmall", 2): -25000.0,
            ("Warehouse", 1): -40000.0,
            ("SingleFamilyHouse", 1): -10000.0,
        }

    def test_by_building_id(self) -> None:
        score = scoring.compute_normalized_score(
            cumulative_return=-30000.0,
            building_type="OfficeSmall",
            task="task1",
            building_id=1,
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
            building_id=1,
        )
        assert score == pytest.approx(-15000.0 / -30000.0)

    def test_missing_building_id_raises(self) -> None:
        with pytest.raises(KeyError, match="No baseline return found"):
            scoring.compute_normalized_score(
                cumulative_return=-1.0,
                building_type="OfficeSmall",
                task="task1",
                building_id=999,
            )

    def test_missing_building_type_raises(self) -> None:
        with pytest.raises(KeyError, match="No baseline returns found"):
            scoring.compute_normalized_score(
                cumulative_return=-1.0,
                building_type="NonExistent",  # type: ignore[arg-type]
                task="task1",
            )

    def test_zero_baseline_returns_raw(self) -> None:
        scoring._baseline_cache = {("ZeroType", 1): 0.0}
        score = scoring.compute_normalized_score(
            cumulative_return=-500.0,
            building_type="ZeroType",  # type: ignore[arg-type]
            task="task1",
            building_id=1,
        )
        assert score == -500.0
