"""Tests for the real CSV-loading path in ``building2building.scoring``."""

from __future__ import annotations

from pathlib import Path

import pytest

from building2building import scoring


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    scoring._cache = None


@pytest.mark.quick
class TestScoringCsvLoader:
    def test_loads_fixture_csv_happy_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_csv_path: Path,
    ) -> None:
        monkeypatch.setattr(scoring, "CSV_PATH", baseline_csv_path)
        loaded = scoring._load()
        assert loaded[("OfficeSmall", "task1", "full_year", "OfficeSmall-0001")] == pytest.approx(
            -30000.0
        )
        assert loaded[("OfficeSmall", "task1", "winter", "OfficeSmall-0001")] == pytest.approx(
            -12000.0
        )
        assert loaded[("Warehouse", "task1", "full_year", "Warehouse-0001")] == pytest.approx(
            -40000.0
        )

    def test_column_rename_raises_clear_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_csv_path: Path,
        tmp_path: Path,
    ) -> None:
        renamed = tmp_path / "baseline_returns_renamed.csv"
        renamed.write_text(
            baseline_csv_path.read_text().replace("reward_mean", "reward"), encoding="utf-8"
        )
        monkeypatch.setattr(scoring, "CSV_PATH", renamed)
        with pytest.raises(KeyError, match="reward_mean"):
            scoring._load()

    def test_missing_row_raises_key_with_tuple_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_csv_path: Path,
    ) -> None:
        monkeypatch.setattr(scoring, "CSV_PATH", baseline_csv_path)
        with pytest.raises(
            KeyError,
            match="No baseline return found.*OfficeSmall.*task1.*summer.*OfficeSmall-0001",
        ):
            scoring.compute_normalized_score(
                cumulative_return=-10000.0,
                building_type="OfficeSmall",
                task="task1",
                run_period="summer",
                building_id="OfficeSmall-0001",
            )
