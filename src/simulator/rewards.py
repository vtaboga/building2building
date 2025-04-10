import numpy as np

def base_reward_function(obs, setpoints=None) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.
    
    Args:
        obs: Dictionary containing observations
        setpoints: Dictionary mapping zones to their setpoint configurations
        
    Returns:
        float: Combined reward (negative values represent penalties)
    """
    # Energy consumption penalty
    power_penalty = obs["energy"]["HVAC_electricity"] + obs["energy"]["HVAC_natural_gas"]
    
    # If no setpoints provided, return just the power penalty
    if setpoints is None:
        return -power_penalty
        
    # Get controlled zones (zones that have setpoints)
    controlled_zones = list(setpoints.keys())
    
    # Calculate temperature tracking error for controlled zones
    temp_error = 0
    target_temp = 21.0  # Target temperature in °C
    
    for zone in controlled_zones:
        current_temp = obs["temperature"][zone]
        temp_error += (current_temp - target_temp) ** 2
        print(f"Zone: {zone}, Current Temp: {current_temp}, Target Temp: {target_temp}, Temp Error: {temp_error}")
    
    # Calculate mean squared error
    if controlled_zones:
        temp_error = temp_error / len(controlled_zones)
    
    # Combine rewards (negative values represent penalties)
    # Equal weighting between temperature tracking and energy consumption
    total_reward = -(temp_error + power_penalty)
    
    return total_reward
