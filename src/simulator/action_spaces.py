import numpy as np
import typing
import gymnasium as gym

def action_transform(act, actuators):
    """Transform raw actions into heating and cooling setpoints.
    
    Args:
        act: Raw actions from the policy
        actuators: Dictionary of actuators with their schedule names
    """
    # Get schedule names from actuators
    schedule_names = list(actuators.keys())
    
    # Ensure we have exactly two schedules (heating and cooling)
    if len(schedule_names) != 2:
        raise ValueError(f"Expected 2 schedule names (heating and cooling), got {len(schedule_names)}")
    
    # Sort to ensure consistent ordering (heating should be first due to alphabetical order)
    schedule_names.sort()
    
    return {
        schedule_names[0]: act[0],  # Heating setpoint (first alphabetically)
        schedule_names[1]: act[1],  # Cooling setpoint (second alphabetically)
    }

def create_action_space(actuators) -> gym.spaces.Box:
    """Create a Gymnasium action space for temperature setpoints."""
    # Verify we have exactly two actuators (heating and cooling)
    if len(actuators) != 2:
        raise ValueError(f"Expected 2 actuators (heating and cooling), got {len(actuators)}")
    
    # Temperature bounds in Celsius
    return gym.spaces.Box(
        np.array([12.0, 12.0]),  # Min heating and cooling temps
        np.array([40.0, 40.0]),  # Max heating and cooling temps
    )
