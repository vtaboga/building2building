import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from building2building.simulator.observation_spaces import extract_ordered_observations


class TrajectoryLogger:
    save_dir: Path
    trajectories: Any
    logger: logging.Logger
    observation_names: list[str]
    total_reward: float

    def __init__(
        self,
        save_dir: Path,
        observation_names: list[str],
        logger: logging.Logger | None = None,
    ):
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
            f"{name} (uncontrolled)"
            if self._is_uncontrolled_zone(name, uncontrolled_zones)
            else name
            for name in self.observation_names
        ]

        # Convert NumPy types to native Python types for JSON serialization
        state_dict = {
            name: float(value) if hasattr(value, "item") else value
            for name, value in zip(names, state)
        }

        self.total_reward += reward

        self.trajectories.append(
            {
                "state": state_dict,
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": float(reward) if hasattr(reward, "item") else reward,
            }
        )

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

        with open(file_path, "w") as f:
            json.dump(self.trajectories, f, indent=4, cls=NumpyEncoder)
        self.logger.info(f"Trajectories saved to {file_path}")


class CustomRescaleAction(gym.wrappers.RescaleAction):
    def __init__(self, env, min_action, max_action):
        """
        Initialize the CustomRescaleAction wrapper.

        Args:
            env: The environment to wrap
            min_action: The minimum value of the new action space
            max_action: The maximum value of the new action space
        """
        super().__init__(env, min_action, max_action)
        self.min_action = min_action
        self.max_action = max_action

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Scale an action from the normalized space (min_action to max_action)
        to the environment's original action space and clip it to stay within bounds.

        Args:
            action: Action in the normalized space (min_action to max_action)

        Returns:
            Action scaled to the environment's original action space and clipped
        """
        # Scale from [min_action, max_action] to [0, 1]
        norm_action = (action - self.min_action) / (self.max_action - self.min_action)

        # Scale from [0, 1] to environment's action space
        scaled_action = (
            norm_action * (self.env.action_space.high - self.env.action_space.low)
            + self.env.action_space.low
        )

        # Clip the action to ensure it stays within the environment's action space
        # This wrapper is usually used in conjunction with a clipping wrapper
        clipped_action = np.clip(
            scaled_action, self.env.action_space.low, self.env.action_space.high
        )

        return clipped_action
