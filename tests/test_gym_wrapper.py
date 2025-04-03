import pytest
import numpy as np
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
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
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
