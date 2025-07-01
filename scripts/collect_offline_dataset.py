"""
Script to collect offline datasets for offline RL training using Hydra configuration.
"""

import os
import json
import numpy as np
import gymnasium as gym
import torch
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from typing import Dict, List
from torch.distributions import Normal

from building2building.simulator.wrappers import NormalizeObservation, CustomRescaleAction
import building2building.simulator


def constant_policy(obs: np.ndarray, heating_setpoint: float, cooling_setpoint: float, 
                   action_space) -> np.ndarray:
    """Simple constant policy for baseline data collection."""
    # The first value is the heating setpoint
    # The second value is the offset from the heating setpoint to the cooling setpoint
    action = np.array([heating_setpoint, cooling_setpoint - heating_setpoint])
    # Normalize each action dimension to [0, 1] for the normalized environment
    action = (action - action_space.low) / (action_space.high - action_space.low)
    return action


def load_dqn_policy(model_path: str, env: gym.Env, device: str = "cpu"):
    """Load trained DQN policy."""
    from building2building.algorithms.online.dqn import QNetwork
    import gymnasium as gym
    
    # Create a vectorized environment for the QNetwork initialization
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    policy = QNetwork(envs, bins_per_dimension=20).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    
    def get_action(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ = policy.get_action(obs_tensor, epsilon=0.0)  # No exploration
        return action
    
    return get_action


def load_ppo_policy(model_path: str, env: gym.Env, device: str = "cpu"):
    """Load trained PPO policy."""
    from building2building.algorithms.online.ppo import Agent
    import gymnasium as gym
    
    # Create a vectorized environment for the Agent initialization
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    policy = Agent(envs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    
    def get_action(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mean = policy.actor_mean(obs_tensor)
            action_logstd = policy.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        return action.cpu().numpy().flatten()
    
    return get_action


def collect_data_from_policy(env: gym.Env, policy_fn, num_episodes: int, 
                           logger: logging.Logger) -> Dict[str, List]:
    """Collect data using any policy function in D4RL format."""
    
    data = {
        'observations': [],
        'next_observations': [],  
        'actions': [],
        'rewards': [],
        'terminals': [],
        'timeouts': []
    }
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action = policy_fn(obs)
            
            # Store current observation and action
            data['observations'].append(obs.copy())
            data['actions'].append(action.copy())
            
            # Take step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store next observation and outcomes
            data['next_observations'].append(next_obs.copy())  
            data['rewards'].append(reward)
            data['terminals'].append(done)
            data['timeouts'].append(truncated)
            
            obs = next_obs
    
        logger.info(f"Completed episode {episode + 1}/{num_episodes}")
    
    return data


def save_dataset(data: Dict[str, List], output_path: str, logger: logging.Logger):
    """Save dataset to file."""
    
    # Convert lists to numpy arrays
    dataset = {}
    for key, values in data.items():
        if key in ['rewards', 'terminals', 'timeouts']:
            dataset[key] = np.array(values).reshape(-1, 1)
        else:
            dataset[key] = np.array(values)
    
    # Log dataset statistics
    n_transitions = len(dataset['observations'])
    total_reward = np.sum(dataset['rewards'])
    mean_reward = np.mean(dataset['rewards'])
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total transitions: {n_transitions}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Mean reward per step: {mean_reward:.4f}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as NPZ format
    np.savez_compressed(output_path, **dataset)
    logger.info(f"Dataset saved to: {output_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for dataset collection with Hydra configuration."""
    
    # Ensure we're using the dataset collection experiment
    if cfg.name != "collect_dataset":
        raise ValueError(f"This script expects collect_dataset experiment, got {cfg.name}")
    
    # Get Hydra's output directory and setup logging
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger(__name__)
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        logger.error(f"Building characteristics file not found: {characteristics_path}")
        raise
    
    # Create environment
    env = gym.make(
        "EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=f"data/weather/{cfg.building.weather}",
        building_characteristics=building_characteristics,
        reward_type=cfg.env.reward_type,
        energy_weight=cfg.env.energy_weight,
    )
    
    # Get action space before wrapping 
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("Environment must have a Box action space")
    action_space = env.action_space
    
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    
    # Set seed
    env.reset(seed=cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Determine policy type and load accordingly
    policy_type = cfg.dataset.get('policy_type', 'baseline')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if policy_type == 'baseline':
        logger.info(f"Starting baseline data collection for {cfg.dataset.num_episodes} episodes")
        policy_fn = lambda obs: constant_policy(obs, cfg.dataset.heating_setpoint, 
                                               cfg.dataset.cooling_setpoint, action_space)
        policy_name = "baseline"
        
    elif policy_type == 'dqn':
        model_path = cfg.dataset.model_path
        logger.info(f"Loading DQN policy from {model_path}")
        policy_fn = load_dqn_policy(model_path, env, device)
        policy_name = "dqn"
        
    elif policy_type == 'ppo':
        model_path = cfg.dataset.model_path
        logger.info(f"Loading PPO policy from {model_path}")
        policy_fn = load_ppo_policy(model_path, env, device)
        policy_name = "ppo"
        
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'baseline', 'dqn', or 'ppo'")
    
    # Collect data
    logger.info(f"Starting {policy_type} data collection for {cfg.dataset.num_episodes} episodes")
    data = collect_data_from_policy(env, policy_fn, cfg.dataset.num_episodes, logger)
    
    # Save dataset to data/offline_dataset directory
    dataset_dir = Path("data/offline_dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / f"{cfg.building.building_id}_{policy_name}_dataset.npz"
    save_dataset(data, str(output_path), logger)
    
    env.close()
    logger.info("Dataset collection completed!")


if __name__ == "__main__":
    main() 