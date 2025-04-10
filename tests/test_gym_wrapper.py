import pytest
import numpy as np
import os
import glob
import shutil
import gymnasium as gym
import sys

print("Python path:", sys.path)  # Debug print
print("Importing simulator package...")  # Debug print
import src.simulator  # This should trigger the registration
print("Available gym envs:", gym.envs.registry.keys())  # Debug print

def test_gym_wrapper():
    # Create environment using gym registration - note the exact ID match
    env = gym.make(
        'EnergyPlus-v0',  
        path_to_building="tests/fixtures/building.epJSON",
        path_to_weather="tests/fixtures/alaska.epw"
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
    eplus_output_dir = "eplus_output"
    if os.path.exists(eplus_output_dir):
        try:
            shutil.rmtree(eplus_output_dir)
        except OSError:
            pass  # Ignore errors if directory can't be deleted
