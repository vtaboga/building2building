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
from building2building.simulator.action_spaces import (
    DualSetpoint,
    get_controllable_setpoints,
)
from building2building.types import BuildingCharacteristics, BuildingConfig
from minergym.ontology import Ontology

b2benv.DataPaths._data_path.set(Path("tests/data").resolve())


@pytest.fixture
def weather_file():
    return (
        b2benv.DataPaths.weather_dir()
        / "USA_VT_Bennington-Morse.State.AP.726166_TMYx.2004-2018.epw"
    )


@pytest.fixture
def building_epjson():
    return b2benv.DataPaths.processed_dir() / "VT/Caledonia/1002000371360.epJSON"


@pytest.fixture
def ont_building(building_epjson):
    return Ontology.from_json(building_epjson)


@pytest.fixture
def building_characteristics(building_epjson):
    return BuildingCharacteristics.load_json(building_epjson.with_suffix(".json"))


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


def test_get_controllable_setpoints(ont_building):
    # Get controllable setpoints
    setpoints_1z = get_controllable_setpoints(ont_building)

    # Verify first building (1z) setpoints
    assert len(setpoints_1z) == 1, "Expected exactly one zone in first building"

    assert "Space 1 ZN" in setpoints_1z, "Expected zone 'Space 0 ZN' in first building"

    zone_setpoints_1z = setpoints_1z["Space 1 ZN"]
    assert zone_setpoints_1z == [
        DualSetpoint(
            heating_actuator="Zone Temperature Control,Temperature Heating Setpoint,Space 1 ZN",
            heating_schedule="Space Type 1 Thermostat 2 Heating Setpoint",
            cooling_actuator="Zone Temperature Control,Temperature Cooling Setpoint,Space 1 ZN",
            cooling_schedule="Space Type 1 Thermostat 2 Cooling Setpoint",
        )
    ]


def test_gym_environment_creation(
    building_epjson, weather_file, building_characteristics
):
    """Test creating and using the gym environment."""

    config = BuildingConfig(
        building_epjson,
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
