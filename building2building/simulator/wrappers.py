import gymnasium as gym
import numpy as np
from typing import Any


class NormalizeObservation(gym.ObservationWrapper):
    """
    Observation wrapper that normalizes observations to a [0, 1] range according to the observation space bounds.
    Values may be outside of this range if they are out of the environment's observation space bounds.
    """
    
    def __init__(self, env: gym.Env, dtype: np.dtype = np.float64):
        """ 
        Args:
            env: The environment to wrap
            dtype: The dtype of the observation space
        """
        super().__init__(env)
        
        # Ensure the observation space is a Box
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Expected observation space to be Box, got {type(env.observation_space)}"
            )

        
        # Store the original observation space bounds
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high
        
        # Handle infinite bounds by replacing them with large finite values
        self.obs_low = np.where(
            np.isinf(self.obs_low), -1e10, self.obs_low
        )
        self.obs_high = np.where(
            np.isinf(self.obs_high), 1e10, self.obs_high
        )
        
        # Calculate the range of the observation space
        self.obs_range = self.obs_high - self.obs_low
        # Avoid division by zero for dimensions with zero range
        self.obs_range = np.where(self.obs_range == 0, 1.0, self.obs_range)
        
        # Set the new observation space to be in the target range
        target_low = np.zeros(self.obs_low.shape, dtype=dtype)
        target_high = np.ones(self.obs_high.shape, dtype=dtype)
        
        self.observation_space = gym.spaces.Box(
            low=target_low, 
            high=target_high, 
            dtype=dtype
        )
    
    def observation(self, observation: Any) -> np.ndarray:
        """
        Normalize the observation to the target range.
        
        Args:
            observation: The original observation from the environment
            
        Returns:
            The normalized observation
        """
        observation = np.asarray(observation, dtype=self.observation_space.dtype)
        normalized_observation = (observation - self.obs_low) / self.obs_range
        return normalized_observation
    
    def denormalize(self, observation: np.ndarray) -> np.ndarray:
        """
        Denormalize the observation to the original range.
        """
        return observation * self.obs_range + self.obs_low
    
        
class CustomRescaleAction(gym.wrappers.RescaleAction):
    def __init__(self, env, min_action=0.0, max_action=1.0):
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
        scaled_action = norm_action * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low
        
        # Clip the action to ensure it stays within the environment's action space
        # This wrapper is usually used in conjunction with a clipping wrapper
        clipped_action = np.clip(
            scaled_action,
            self.env.action_space.low,
            self.env.action_space.high
        )
        
        return clipped_action