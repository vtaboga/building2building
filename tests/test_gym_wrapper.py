import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.simulator.action_spaces import (
    DualSetpoint,
    get_controllable_setpoints,
)
from building2building.store import LocalFile, realize
from building2building.types import BaseRewardConfig, BuildingConfig
from minergym.ontology import Ontology

logging.basicConfig(level=logging.DEBUG)

here = Path(__file__).parent.resolve()


@pytest.fixture
def weather_file():
    return realize(
        STORE_PATH.get(),
        LocalFile(
            here / "data/USA_VT_Bennington-Morse.State.AP.726166_TMYx.2004-2018.epw",
        ),
    )


@pytest.fixture
def building_epjson():
    x = LocalFile(
        here / "data/1002000371360.idf",
    )

    return realize(
        STORE_PATH.get(),
        create_complete_pipeline(x, energyplus_path(), src_version="9.4.0"),
    )


@pytest.fixture
def ont_building(building_epjson):
    return Ontology.from_json(building_epjson)


@pytest.fixture
def building_config(building_epjson, weather_file):
    config = BuildingConfig(
        building_epjson,
        weather_file,
        BaseRewardConfig(1000),
        1.0,
        Path("eplus_output"),
        3,
    )


def test_get_controllable_setpoints(ont_building):
    # Get controllable setpoints
    setpoints_1z = get_controllable_setpoints(ont_building)

    # Verify first building (1z) setpoints
    assert len(setpoints_1z) == 1, "Expected exactly one zone in first building"

    assert "Space 1 ZN" in setpoints_1z, "Expected zone 'Space 0 ZN' in first building"

    zone_setpoints_1z = setpoints_1z["Space 1 ZN"]
    assert zone_setpoints_1z == [
        DualSetpoint(
            name="Space Type 1 Thermostat 2",
            heating_actuator="Zone Temperature Control,Temperature Heating Setpoint,Space 1 ZN",
            heating_schedule="Space Type 1 Thermostat 2 Heating Setpoint",
            cooling_actuator="Zone Temperature Control,Temperature Cooling Setpoint,Space 1 ZN",
            cooling_schedule="Space Type 1 Thermostat 2 Cooling Setpoint",
        )
    ]


def test_gym_environment_creation(building_epjson, weather_file):
    """Test creating and using the gym environment."""

    config = BuildingConfig(
        building_epjson,
        weather_file,
        BaseRewardConfig(1000),
        1.0,
        Path("help"),
        3,
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
