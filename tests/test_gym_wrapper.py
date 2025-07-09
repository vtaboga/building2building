import glob
import json
import os
import shutil
import sys
from pathlib import Path

import building2building.env as b2benv
import building2building.simulator
import gymnasium as gym
import numpy as np
import pytest
from building2building.types import BuildingCharacteristics, BuildingConfig


def test_gym_wrapper():
    # Create environment using gym registration - note the exact ID match

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

    env = gym.make(
        "EnergyPlus-v0",
        building_config=building_config,
    )

    # Reset environment and get initial observation
    obs, info = env.reset()

    # Verify observation shape matches what we expect
    assert isinstance(obs, np.ndarray)

    # Get action space from environment
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box)

    # Run a few simulation steps with random actions
    for _ in range(5):
        action = action_space.sample()  # Use random actions from action space

        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)

        # Basic assertions to verify step output
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        if terminated or truncated:
            obs, info = env.reset()

    # Clean up
    env.close()

    # Delete the eplus_output directory and all its contents

    if eplus_output_dir.exists():
        try:
            shutil.rmtree(eplus_output_dir)
        except OSError:
            pass  # Ignore errors if directory can't be deleted
