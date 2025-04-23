import os
import json
import logging

class TrajectoryLogger:
    def __init__(self, save_dir, logger=None):
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

    def log(self, state, action, reward):
        """
        Log a single step in the trajectory.

        Args:
            state: The observed state.
            action: The action taken.
            reward: The reward received.
        """
        self.trajectories.append({
            "state": state.tolist() if hasattr(state, 'tolist') else state,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "reward": reward
        })

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
