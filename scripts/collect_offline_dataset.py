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

from building2building.simulator.wrappers import NormalizeObservation, CustomRescaleAction
import building2building.simulator


def constant_policy(obs: np.ndarray, heating_setpoint: float, cooling_setpoint: float, 
                   action_space) -> np.ndarray:
    """Simple constant policy for baseline data collection."""
    action = np.array([heating_setpoint, cooling_setpoint - heating_setpoint])
    # Normalize to [0, 1] range
    action = (action - action_space.low) / (action_space.high - action_space.low)
    return action


def collect_baseline_data(env: gym.Env, num_episodes: int, heating_setpoint: float,
                         cooling_setpoint: float, logger: logging.Logger) -> Dict[str, List]:
    """Collect data using constant baseline policy in D4RL format."""
    
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
            action = constant_policy(obs, heating_setpoint, cooling_setpoint, env.action_space)
            
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
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    
    # Set seed
    env.reset(seed=cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Collect data
    logger.info(f"Starting baseline data collection for {cfg.dataset.num_episodes} episodes")
    
    data = collect_baseline_data(
        env, 
        cfg.dataset.num_episodes,
        cfg.dataset.heating_setpoint,
        cfg.dataset.cooling_setpoint,
        logger
    )
    
    # Save dataset
    output_path = output_dir / f"{cfg.building.building_id}_dataset.npz"
    save_dataset(data, str(output_path), logger)
    
    env.close()
    logger.info("Dataset collection completed!")


if __name__ == "__main__":
    main() 