import numpy as np
import typing
import gymnasium as gym


def extract_ordered_observations(obs):
    """Extract specific values from the raw observation dictionary in a consistent order.
    
    The returned list contains in order:
    - Zone Air Temperatures (alphabetically ordered by zone name)
    - Outdoor Air Temperature
    - Outdoor Air Relative Humidity
    - Current Time of Day
    - Day of Year
    - HVAC Electricity Consumption
    - HVAC Natural Gas Consumption
    """
    # Get sorted list of zones to ensure consistent order - zones are already decoded
    zones = sorted(obs["temperature"].keys())
    
    return (
        # Zone temperatures (in sorted order)
        [obs["temperature"][zone] for zone in zones] +
        # Weather variables
        [
            obs["weather"]["drybulb_temp"],
            obs["weather"]["relative_humidity"]
        ] +
        # Time variables
        [
            obs["time"]["current_time"],
            obs["time"]["day_of_year"]
        ] +
        # Energy variables 
        [
            obs["energy"]["HVAC_electricity"],
            obs["energy"]["HVAC_natural_gas"]
        ]
    )


def observation_transform(obs):
    """Transform raw observations into a numpy array by extracting and ordering
    relevant values.
    """
    obs = np.array(extract_ordered_observations(obs))
    return obs


def create_observation_space(obs_template: typing.Dict[str, typing.Any]) -> gym.spaces.Box:
    """Create the observation space with appropriate bounds for each variable type.
    
    The observation space contains in order:
    - Zone Air Temperatures (one per zone) [-50°C, 50°C]
    - Outdoor Air Temperature [-50°C, 50°C]
    - Outdoor Air Relative Humidity [0%, 100%]
    - Current Time of Day [0, 24]
    - Day of Year [1, 366]
    - HVAC Electricity Consumption [0, inf]
    - HVAC Natural Gas Consumption [0, inf]
    
    Args:
        obs_template: Dictionary containing the observation template
        
    Returns:
        gym.spaces.Box: The observation space with appropriate bounds
    """
    observation_mock = extract_ordered_observations(obs_template)
    num_obs = len(observation_mock)
    
    # Create appropriate bounds for each observation type
    low_bounds = np.zeros(num_obs)
    high_bounds = np.ones(num_obs) * float('inf')
    
    # Get index for each observation type
    num_zones = len(obs_template["temperature"])
    idx = 0
    
    # Temperature bounds indoor
    temp_end_idx = idx + num_zones  # zones + outdoor
    low_bounds[idx:temp_end_idx] = 10.0  # Very cold
    high_bounds[idx:temp_end_idx] = 40.0  # Very hot
    idx = temp_end_idx

    # Temperature bounds outdoor
    low_bounds[idx] = -50.0  
    high_bounds[idx] = 50.0  
    idx += 1

    # Humidity bounds 
    low_bounds[idx] = 0.0  # 0%
    high_bounds[idx] = 100.0  # 100%
    idx += 1
    
    # Time bounds
    low_bounds[idx] = 0.0  # 0 hours
    high_bounds[idx] = 24.0  # 24 hours
    idx += 1
    
    # Day of year bounds
    low_bounds[idx] = 1.0  # Day 1
    high_bounds[idx] = 366.0  # Day 366 (leap year)
    idx += 1
    
    # Energy bounds (already set to [0, inf])
    
    return gym.spaces.Box(low_bounds, high_bounds)

