import json
import shutil
import unittest
from pathlib import Path

import pytest
from building2building.algorithms.online.baselines import run_constant_baseline
from building2building.types import BuildingCharacteristics, BuildingConfig


def test_run_constant_baseline_basic():
    pytest.skip("takes too long")

    output_dir = Path("tests/temp_output")
    output_dir.mkdir(exist_ok=True)

    # Setup test paths
    building_id = "1003000512784"
    building_path = Path(f"tests/fixtures/processed_buildings/{building_id}.epJSON")
    weather_path = Path("tests/fixtures/weather/weather_vt_1.epw")
    characteristics_path = Path(
        f"tests/fixtures/processed_buildings/{building_id}.json"
    )
    fixture_trajectory = f"tests/fixtures/results/{building_id}_trajectory.json"

    # Load building characteristics
    building_characteristics = BuildingCharacteristics.load_json(characteristics_path)

    eplus_output_dir = output_dir / "eplus_output"

    conf = BuildingConfig(
        building_path,
        weather_path,
        building_characteristics,
        "base",
        1.0,
        eplus_output_dir,
    )

    # Run baseline
    run_constant_baseline(
        "EnergyPlus-v0",
        conf,
        heating_setpoint=21.0,
        cooling_setpoint=24.0,
        seed=42,
        results_dir=output_dir,
    )

    # Load and compare trajectories
    with open(fixture_trajectory, "r") as f:
        expected_trajectory = json.load(f)
    with open(output_dir / "trajectories/trajectories.json", "r") as f:
        actual_trajectory = json.load(f)

    assert actual_trajectory == expected_trajectory, (
        "Trajectory does not match expected fixture"
    )
