import sys
import os
from pathlib import Path
import json
import logging
import gymnasium as gym
import numpy as np
import pickle
from src.simulator.observation_spaces import extract_ordered_observations
from src.simulator.config_manager import setup_energyplus_path, find_energyplus_path


def get_energyplus_env():
    """
    Get the EnergyPlus path in a format suitable for environment variables.
    Returns a tuple of (variable_name, path).
    """
    path = find_energyplus_path()
    if path:
        return "ENERGYPLUS_PATH", path
    return None

# Call setup_energyplus_path when the module is imported
# This ensures EnergyPlus can be found before any imports that depend on it
setup_energyplus_path()


class TrajectoryLogger:
    def __init__(self, save_dir, observation_names, logger=None):
        """
        Initialize the TrajectoryLogger.

        Args:
            save_dir (str): Directory where trajectories will be saved.
            logger (logging.Logger): Logger for logging messages.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.trajectories = []
        self.logger = logger or logging.getLogger(__name__)
        self.observation_names = observation_names
        self.total_reward = 0

    def log(self, state, action, reward, controlled_zones, uncontrolled_zones):
        """
        Log a single step in the trajectory.

        Args:
            state: The observed state.
            action: The action taken.
            reward: The reward received.
            controlled_zones: The zones that are controlled.
            uncontrolled_zones: The zones that are not controlled.
        """

        names = [
            f"{name} (uncontrolled)" if self._is_uncontrolled_zone(name, uncontrolled_zones) else name
            for name in self.observation_names
        ]

        # Convert NumPy types to native Python types for JSON serialization
        state_dict = {name: float(value) if hasattr(value, 'item') else value 
                     for name, value in zip(names, state)}

        self.total_reward += reward

        self.trajectories.append({
            "state": state_dict,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "reward": float(reward) if hasattr(reward, 'item') else reward
        })

    @staticmethod
    def _is_uncontrolled_zone(name, uncontrolled_zones):
        if "Zone Temperature" not in name:
            return False
        return any([zone in name for zone in uncontrolled_zones])

    @property
    def trajectories_path(self):
        return os.path.join(self.save_dir, "trajectories.json")

    def save(self, filename="trajectories.json"):
        """
        Save the logged trajectories to a file.

        Args:
            filename (str): The name of the file to save the trajectories.
        """
        file_path = os.path.join(self.save_dir, filename)
        
        # Create a custom encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(file_path, 'w') as f:
            json.dump(self.trajectories, f, indent=4, cls=NumpyEncoder)
        self.logger.info(f"Trajectories saved to {file_path}")


