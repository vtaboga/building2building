"""Pins the ``prepare_building`` end-to-end pipeline contract.

Asserts that ``prepare_building`` converts a raw IDF fixture through the full
pipeline (upgrade → convert → add HVAC meters → add outdoor-air variables →
set timestep → set run period) and produces a valid epJSON with the expected
output variables, timestep, and run-period dates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import prepare_building
from building2building.store import realize

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline_idfs"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("idf_name", "timesteps_per_hour", "expect_schedule_file"),
    [
        ("vav_multizone.idf", 6, False),
        ("sfh_with_schedule_file.idf", 4, True),
    ],
)
def test_prepare_building_end_to_end(
    idf_name: str, timesteps_per_hour: int, expect_schedule_file: bool
) -> None:
    epjson_path = realize(
        STORE_PATH.get(),
        prepare_building(
            input_file=FIXTURES_DIR / idf_name,
            energyplus_path=energyplus_path(),
            src_version="24.1.0",
            timesteps_per_hour=timesteps_per_hour,
        ),
    )
    epjson = json.loads(epjson_path.read_text())

    output_meter = epjson.get("Output:Meter", {})
    assert "Output:Meter:ElectricityHVAC" in output_meter
    assert "Output:Meter:NaturalGasHVAC" in output_meter

    outdoor_variable_names = {
        item.get("variable_name", "")
        for item in epjson.get("Output:Variable", {}).values()
    }
    assert "Site Outdoor Air Drybulb Temperature" in outdoor_variable_names
    assert "Site Outdoor Air Humidity Ratio" in outdoor_variable_names

    timestep_values = {
        item.get("number_of_timesteps_per_hour")
        for item in epjson.get("Timestep", {}).values()
    }
    assert timestep_values == {timesteps_per_hour}

    run_period = epjson["RunPeriod"]["Run Period 1"]
    assert run_period["begin_month"] == 1
    assert run_period["begin_day_of_month"] == 1
    assert run_period["end_month"] == 12
    assert run_period["end_day_of_month"] == 31

    # Ensure the generated epJSON can always be serialized/parsed.
    reparsed = json.loads(json.dumps(epjson))
    assert isinstance(reparsed, dict)

    if expect_schedule_file:
        schedule_files = epjson.get("Schedule:File", {})
        assert schedule_files, "Expected at least one Schedule:File entry in SFH fixture"
        for sched_obj in schedule_files.values():
            assert isinstance(sched_obj.get("file_name", ""), str)
