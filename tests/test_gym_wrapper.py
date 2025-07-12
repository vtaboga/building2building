import glob
import json
import os
import shutil
import sys
from pathlib import Path

import building2building.env as b2benv
import gymnasium as gym
import numpy as np
import pytest
from building2building.simulator.action_spaces import get_controllable_setpoints
from building2building.types import BuildingCharacteristics, BuildingConfig
from minergym.ontology import Ontology


@pytest.fixture
def fixtures_dir():
    return Path("tests/fixtures/processed_buildings")


@pytest.fixture
def weather_file():
    return "tests/fixtures/weather/weather_vt_1.epw"


@pytest.fixture
def building_1z_epjson(fixtures_dir):
    return fixtures_dir / "1003000523385.epJSON"


@pytest.fixture
def building_2z_epjson(fixtures_dir):
    return fixtures_dir / "1003000529058.epJSON"


@pytest.fixture
def ont_1z(building_1z_epjson):
    return Ontology.from_json(building_1z_epjson)


@pytest.fixture
def ont_2z(building_2z_epjson):
    return Ontology.from_json(building_2z_epjson)


@pytest.fixture
def building_characteristics(building_1z_epjson):
    return BuildingCharacteristics.load_json(building_1z_epjson.with_suffix(".json"))


def test_gym_wrapper():
    # Create environment using gym registration - note the exact ID match
    pytest.skip("for now")
    eplus_output_dir = Path("eplus_output")

    building_characteristics = BuildingCharacteristics.load_json(
        Path("tests/fixtures/building.json")
    )

    building_config = BuildingConfig(
        Path("tests/fixtures/building.epJSON"),
        Path("tests/fixtures/alaska.epw"),
        building_characteristics,
        "base",
        1.0,
        eplus_output_dir,
    )

    if b2benv.ENERGYPLUS_PATH.get(None) is None:
        pytest.skip("Energyplus not installed or not setup.")


def test_get_controllable_setpoints(ont_1z, ont_2z):
    # Get controllable setpoints
    setpoints_1z = get_controllable_setpoints(ont_1z)
    setpoints_2z = get_controllable_setpoints(ont_2z)

    # Verify first building (1z) setpoints
    assert len(setpoints_1z) == 1, "Expected exactly one zone in first building"

    assert "Space 0 ZN" in setpoints_1z, "Expected zone 'Space 0 ZN' in first building"

    zone_setpoints_1z = setpoints_1z["Space 0 ZN"]
    assert len(zone_setpoints_1z) == 2, (
        "Expected exactly two setpoints in first building"
    )

    # Check heating setpoint for first building
    heating_1z = next(
        sp for sp in zone_setpoints_1z if sp["setpoint_type"] == "heating"
    )
    assert heating_1z["schedule_name"] == "Space Type 1 Thermostat 1 Heating Setpoint"

    assert heating_1z["control_type"] == "DualSetpoint"
    assert (
        heating_1z["actuator_key"]
        == "Zone Temperature Control,Temperature Heating Setpoint,Space 0 ZN"
    )

    # Check cooling setpoint for first building
    cooling_1z = next(
        sp for sp in zone_setpoints_1z if sp["setpoint_type"] == "cooling"
    )
    assert cooling_1z["schedule_name"] == "Space Type 1 Thermostat 1 Cooling Setpoint"
    assert cooling_1z["control_type"] == "DualSetpoint"
    assert (
        cooling_1z["actuator_key"]
        == "Zone Temperature Control,Temperature Cooling Setpoint,Space 0 ZN"
    )

    # Verify second building (2z) setpoints
    assert len(setpoints_2z) == 1, "Expected exactly one zone in second building"

    assert "Space 4 ZN" in setpoints_2z, "Expected zone 'Space 4 ZN' in second building"

    zone_setpoints_2z = setpoints_2z["Space 4 ZN"]
    assert len(zone_setpoints_2z) == 2, (
        "Expected exactly two setpoints in second building"
    )

    # Check heating setpoint for second building
    heating_2z = next(
        sp for sp in zone_setpoints_2z if sp["setpoint_type"] == "heating"
    )
    assert heating_2z["schedule_name"] == "Space Type 1 Thermostat 5 Heating Setpoint"
    assert heating_2z["control_type"] == "DualSetpoint"
    assert (
        heating_2z["actuator_key"]
        == "Zone Temperature Control,Temperature Heating Setpoint,Space 4 ZN"
    )

    # Check cooling setpoint for second building
    cooling_2z = next(
        sp for sp in zone_setpoints_2z if sp["setpoint_type"] == "cooling"
    )
    assert cooling_2z["schedule_name"] == "Space Type 1 Thermostat 5 Cooling Setpoint"
    assert cooling_2z["control_type"] == "DualSetpoint"
    assert (
        cooling_2z["actuator_key"]
        == "Zone Temperature Control,Temperature Cooling Setpoint,Space 4 ZN"
    )


def test_gym_environment_creation(
    building_1z_epjson, weather_file, building_characteristics
):
    """Test creating and using the gym environment."""

    config = BuildingConfig(
        building_1z_epjson,
        weather_file,
        building_characteristics,
        "base",
        1.0,
        Path("help"),
    )

    env = gym.make("EnergyPlus-v0", building_config=config)

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(env.action_space, gym.spaces.Box)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
