"""Pins the ``patch_epjson_run_period`` contract.

Asserts that patching an epJSON file rewrites both the ``RunPeriod`` dates and
the ``Schedule:File`` paths to point at the correct seasonal CSV, and that
missing ``RunPeriod`` objects and the ``"summer"`` period alias are handled
correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from building2building.api import _patch_epjson_run_period
from building2building.types import RunPeriodConfig


def _write_epjson(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


@pytest.mark.quick
def test_patch_epjson_run_period_rewrites_schedule_files_and_dates(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_epjson = src_dir / "building.epjson"
    dst_epjson = dst_dir / "building.epjson"

    relative_path = "schedules/profile.csv"
    absolute_path = str((tmp_path / "abs_profile.csv").resolve())
    _write_epjson(
        src_epjson,
        {
            "RunPeriod": {"Run Period 1": {"begin_month": 2, "begin_day_of_month": 5}},
            "Schedule:File": {
                "Relative Schedule": {"file_name": relative_path},
                "Absolute Schedule": {"file_name": absolute_path},
            },
        },
    )
    dst_epjson.parent.mkdir(parents=True, exist_ok=True)

    _patch_epjson_run_period(src_epjson, dst_epjson, RunPeriodConfig.from_name("winter"))
    patched = json.loads(dst_epjson.read_text())

    run_period = patched["RunPeriod"]["Run Period 1"]
    assert run_period["begin_month"] == 1
    assert run_period["begin_day_of_month"] == 1
    assert run_period["end_month"] == 3
    assert run_period["end_day_of_month"] == 31

    scheds = patched["Schedule:File"]
    assert scheds["Relative Schedule"]["file_name"] == str((src_dir / relative_path).resolve())
    assert scheds["Absolute Schedule"]["file_name"] == absolute_path


@pytest.mark.quick
def test_patch_epjson_run_period_handles_summer_and_missing_runperiod(tmp_path: Path) -> None:
    src_epjson = tmp_path / "source" / "building.epjson"
    dst_epjson = tmp_path / "output" / "building.epjson"
    _write_epjson(src_epjson, {"Building": {"Example": {}}})
    dst_epjson.parent.mkdir(parents=True, exist_ok=True)

    _patch_epjson_run_period(src_epjson, dst_epjson, RunPeriodConfig.from_name("summer"))
    patched = json.loads(dst_epjson.read_text())

    run_period = patched["RunPeriod"]["Run Period 1"]
    assert run_period["begin_month"] == 6
    assert run_period["begin_day_of_month"] == 1
    assert run_period["end_month"] == 8
    assert run_period["end_day_of_month"] == 31

    # Contract for the fabricated default branch.
    assert run_period["begin_year"] == 2023
    assert run_period["end_year"] == 2023
    assert run_period["apply_weekend_holiday_rule"] == "No"
