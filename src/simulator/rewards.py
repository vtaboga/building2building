import numpy as np

def base_reward_function(obs, setpoints=None, building_characteristics=None, energy_weight=1.0) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.
    
    Args:
        obs: Dictionary containing observations
        setpoints: Dictionary mapping zones to their setpoint configurations
        building_characteristics: Dictionary containing building characteristics
        energy_weight: Weight for the energy consumption penalty
    Returns:
        float: Combined reward (negative values represent penalties)
    """

    # Energy consumption penalty
    energy_penalty = obs["energy"]["HVAC_electricity"] + obs["energy"]["HVAC_natural_gas"]
    energy_penalty /= 3600  # Convert to Wh

    # Divide energy consumption by the building area if available
    if building_characteristics and 'area' in building_characteristics:
        energy_penalty /= building_characteristics['area']
    
    # If no setpoints provided, return just the power penalty
    if setpoints is None:
        return -energy_penalty
        
    # Get controlled zones (zones that have setpoints)
    controlled_zones = list(setpoints.keys())
    
    # Calculate temperature tracking error for controlled zones
    temp_error = 0
    target_temp = 21.0  # Target temperature in °C
    
    for zone in controlled_zones:
        current_temp = obs["temperature"][zone]
        temp_error += (current_temp - target_temp) ** 2
    
    # Calculate mean squared error
    if controlled_zones:
        temp_error = temp_error / len(controlled_zones)
    
    # Combine rewards (negative values represent penalties)
    # Equal weighting between temperature tracking and energy consumption
    total_reward = -(temp_error + energy_weight * energy_penalty)
    
    return total_reward


def barrier_reward_function(obs, setpoints=None, building_characteristics=None) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.
    
    Args:
        obs: Dictionary containing observations
        setpoints: Dictionary mapping zones to their setpoint configurations
        building_characteristics: Dictionary containing building characteristics
        
    Returns:
        float: Combined reward (negative values represent penalties)
    """

    # Energy consumption penalty
    energy_penalty = obs["energy"]["HVAC_electricity"] + obs["energy"]["HVAC_natural_gas"]
    energy_penalty /= 3600  # Convert to Wh

    # Divide energy consumption by the building area if available
    if building_characteristics and 'area' in building_characteristics:
        energy_penalty /= building_characteristics['area']
    
    # If no setpoints provided, return just the power penalty
    if setpoints is None:
        return -energy_penalty
        
    # Get controlled zones (zones that have setpoints)
    controlled_zones = list(setpoints.keys())
    
    # Calculate temperature tracking error for controlled zones
    temp_error = 0
    target_temp = 21.0  # Target temperature in °C
    delta_temp = 1.5  # Deadband in °C
    penalty = 1000  # Penalty for temperature error outside of deadband
    
    for zone in controlled_zones:
        current_temp = obs["temperature"][zone]
        if abs(current_temp - target_temp) > delta_temp:
            temp_error += penalty
    
    # Calculate mean squared error
    if controlled_zones:
        temp_error = temp_error / len(controlled_zones)
    
    # Combine rewards (negative values represent penalties)
    total_reward = -(temp_error + energy_penalty)
    
    return total_reward
