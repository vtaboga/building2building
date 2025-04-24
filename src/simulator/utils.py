import sys
import os
from pathlib import Path
import json
import logging
from src.simulator.observation_spaces import extract_ordered_observations


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

        state_dict = {name: value for name, value in zip(names, state)}

        self.trajectories.append({
            "state": state_dict,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "reward": reward
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
        with open(file_path, 'w') as f:
            json.dump(self.trajectories, f, indent=4)
        self.logger.info(f"Trajectories saved to {file_path}")


def find_energyplus_path():
    """
    Find the EnergyPlus installation path by checking common locations
    and environment variables.
    """
    # Check environment variable first
    if 'ENERGYPLUS_PATH' in os.environ:
        return os.environ['ENERGYPLUS_PATH']
    
    # Common installation paths
    possible_paths = [
        # Container/Linux path
        '/usr/local/EnergyPlus-24-1-0',
        # Default Linux path
        '/usr/local/energy-plus-24-1-0',
        # Default macOS path
        '/Applications/EnergyPlus-24-1-0',
        # Default Windows path
        'C:\\EnergyPlus-24-1-0',
        # Local development path
        str(Path.home() / 'EnergyPlus-24-1-0'),
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    raise RuntimeError(
        "EnergyPlus installation not found. Please either:\n"
        "1. Set ENERGYPLUS_PATH environment variable\n"
        "2. Install EnergyPlus in one of the standard locations:\n"
        f"   {possible_paths}"
    )

def get_energyplus_env():
    """
    Get the EnergyPlus path in a format suitable for environment variables.
    Returns a tuple of (variable_name, path).
    """
    try:
        path = find_energyplus_path()
        return "ENERGYPLUS_PATH", path
    except RuntimeError as e:
        return None

def setup_energyplus_path():
    """
    Setup the path to EnergyPlus Python API.
    This should be called before any EnergyPlus-related imports.
    """
    energyplus_path = find_energyplus_path()
    
    if energyplus_path not in sys.path:
        sys.path.append(energyplus_path)
        print(f"Added EnergyPlus path: {energyplus_path}")

# Call it when the module is imported
setup_energyplus_path()

