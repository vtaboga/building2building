import numpy as np

def base_reward_function(obs) -> float:
    """Calculate a reward term from a "raw observation"."""
    
    print(f"Obs: {obs}")
    power_penalty = obs["energy"]["HVAC_electricity"] + obs["energy"]["HVAC_natural_gas"]
    return -power_penalty
