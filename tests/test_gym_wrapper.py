import pytest
import numpy as np
import os
import glob
import shutil
from src.simulator.create_simulator import create_simulator

def test_gym_wrapper():
    # Initialize simulator with test files
    env = create_simulator(
        path_to_building="tests/fixtures/building.epJSON",
        path_to_weather="tests/fixtures/alaska.epw"
    )
    
    # Reset environment and get initial observation
    obs, info = env.reset()
    
    # Verify observation shape matches what we expect
    assert isinstance(obs, np.ndarray)
    
    # Run a few simulation steps
    for _ in range(5):
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step([25.0])
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
    eplus_output_dir = "eplus_output"
    if os.path.exists(eplus_output_dir):
        try:
            shutil.rmtree(eplus_output_dir)
        except OSError:
            pass  # Ignore errors if directory can't be deleted
