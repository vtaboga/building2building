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


class CustomRescaleAction(gym.wrappers.RescaleAction):

    def __init__(self, env, min_action, max_action):
        super().__init__(env, min_action, max_action)

    def scale_action(self, action):
        scaled_action = (action + 1.0) / 2.0 * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low
        return scaled_action


class CustomNormalizeObservation(gym.Wrapper):
    """Custom normalization wrapper with special handling for cyclical variables,
    state saving and denormalization support.
    """
    
    def __init__(self, env, epsilon=1e-8, cyclical_indices=None, cyclical_max_values=None):
        """
        Initialize the wrapper with custom handling for cyclical variables.
        
        Args:
            env: The environment to wrap
            epsilon: Small constant to avoid division by zero
            cyclical_indices: List of indices for cyclical variables that should be normalized differently
            cyclical_max_values: List of maximum values for each cyclical variable (used for normalization)
        """
        super().__init__(env)
        
        # Try to get num_envs and is_vector_env attributes
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False
        
        # Default cyclical indices and max values if none provided
        if cyclical_indices is None:
            # For EnergyPlusEnvironment, we can determine indices from observation_names
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'observation_names'):
                obs_names = env.unwrapped.observation_names
                # Find indices for cyclical variables
                time_idx = None
                day_idx = None
                for i, name in enumerate(obs_names):
                    if "Time of Day" in name:
                        time_idx = i
                    elif "Day of Year" in name:
                        day_idx = i
                
                if time_idx is not None and day_idx is not None:
                    self.cyclical_indices = [time_idx, day_idx]
                else:
                    # Fallback to inference based on observation structure
                    num_zones = sum(1 for name in obs_names if 'Zone Temperature' in name)
                    self.cyclical_indices = [num_zones + 2, num_zones + 3]
            else:
                raise ValueError("Observation names not found in environment")
        else:
            self.cyclical_indices = cyclical_indices
            
        # Default max values for time (24) and day of year (365)
        if cyclical_max_values is None:
            self.cyclical_max_values = [24.0, 365.0]
        else:
            self.cyclical_max_values = cyclical_max_values
        
        # Validate
        assert len(self.cyclical_indices) == len(self.cyclical_max_values), \
            "Must provide same number of indices and max values for cyclical variables"
        
        # Initialize RunningMeanStd for non-cyclical variables
        if self.is_vector_env:
            obs_shape = self.single_observation_space.shape
        else:
            obs_shape = self.observation_space.shape
            
        # Create a mask for non-cyclical variables
        self.obs_shape = obs_shape
        self.non_cyclical_mask = np.ones(self.obs_shape, dtype=bool)
        for idx in self.cyclical_indices:
            self.non_cyclical_mask[idx] = False
        
        # Count non-cyclical elements to initialize mean and var with correct shape
        non_cyclical_count = np.sum(self.non_cyclical_mask)
        
        # Initialize RunningMeanStd with the correct shape for non-cyclical variables
        self.obs_rms = RunningMeanStd(shape=(non_cyclical_count,))
        self.epsilon = epsilon
    
    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.observation(obs)
        else:
            obs = self.observation(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.observation(obs), info
        else:
            return self.observation(np.array([obs]))[0], info
    
    def observation(self, observation):
        """
        Normalize the observation with special handling for cyclical variables.
        
        Args:
            observation: The original observation from the environment
            
        Returns:
            Normalized observation with cyclical variables handled separately
        """
        # Convert to numpy array if needed
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Make a copy to avoid modifying the original
        obs_copy = observation.copy().astype(np.float64)
        
        # Extract cyclical and non-cyclical parts
        cyclical_mask = np.zeros(observation.shape[-1], dtype=bool)
        for idx in self.cyclical_indices:
            cyclical_mask[idx] = True
        
        # Handle batch dimension if present
        if len(observation.shape) > 1:
            # For batched observations
            non_cyclical_parts = obs_copy[:, ~cyclical_mask]
            
            # Update running statistics only for non-cyclical parts
            self.obs_rms.update(non_cyclical_parts)
            
            # Normalize non-cyclical parts using running statistics
            normalized_non_cyclical = ((non_cyclical_parts - self.obs_rms.mean) / 
                                     (np.sqrt(self.obs_rms.var) + self.epsilon))
            
            # Handle cyclical variables by division by max value
            result = obs_copy.copy()
            for i, idx in enumerate(self.cyclical_indices):
                max_val = self.cyclical_max_values[i]
                result[:, idx] = obs_copy[:, idx] / max_val
            
            # Combine the results
            result[:, ~cyclical_mask] = normalized_non_cyclical
            
        else:
            # For single observations
            non_cyclical_parts = obs_copy[~cyclical_mask]
            
            # Update running statistics only for non-cyclical parts
            self.obs_rms.update(non_cyclical_parts.reshape(1, -1))
            
            # Normalize non-cyclical parts using running statistics
            normalized_non_cyclical = ((non_cyclical_parts - self.obs_rms.mean) / 
                                     (np.sqrt(self.obs_rms.var) + self.epsilon))
            
            # Handle cyclical variables by division by max value
            result = obs_copy.copy()
            for i, idx in enumerate(self.cyclical_indices):
                max_val = self.cyclical_max_values[i]
                result[idx] = obs_copy[idx] / max_val
            
            # Combine the results
            result[~cyclical_mask] = normalized_non_cyclical
        
        return result
    
    def denormalize(self, normalized_obs):
        """
        Convert normalized observations back to original scale, handling cyclical variables separately.
        
        Args:
            normalized_obs: The normalized observation to be denormalized
            
        Returns:
            The denormalized observation
        """
        # Convert to numpy array if needed
        if not isinstance(normalized_obs, np.ndarray):
            normalized_obs = np.array(normalized_obs)
        
        # Make a copy to avoid modifying the original
        result = normalized_obs.copy()
        
        # Extract cyclical and non-cyclical parts
        cyclical_mask = np.zeros(normalized_obs.shape, dtype=bool)
        for idx in self.cyclical_indices:
            cyclical_mask[idx] = True
        
        # Denormalize non-cyclical parts
        result[~cyclical_mask] = (normalized_obs[~cyclical_mask] * 
                                 (np.sqrt(self.obs_rms.var) + self.epsilon) + 
                                 self.obs_rms.mean)
        
        # Denormalize cyclical variables
        for i, idx in enumerate(self.cyclical_indices):
            max_val = self.cyclical_max_values[i]
            result[idx] = normalized_obs[idx] * max_val
        
        return result
    
    def save_running_state(self, path):
        """
        Save the current normalization statistics to a file.
        
        Args:
            path: Path to save the normalization state
        """
        state = {
            'mean': self.obs_rms.mean,
            'var': self.obs_rms.var,
            'count': self.obs_rms.count,
            'epsilon': self.epsilon,
            'cyclical_indices': self.cyclical_indices,
            'cyclical_max_values': self.cyclical_max_values
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_running_state(self, path):
        """
        Load normalization statistics from a file.
        
        Args:
            path: Path to load the normalization state from
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.obs_rms.mean = state['mean']
        self.obs_rms.var = state['var']
        self.obs_rms.count = state['count']
        self.epsilon = state['epsilon']
        self.cyclical_indices = state['cyclical_indices']
        self.cyclical_max_values = state['cyclical_max_values']


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
