"""
Script to collect offline datasets for offline RL training.
Can collect data from trained policies, baselines, or mixed datasets.
"""

import os
import json
import argparse
import numpy as np
import gymnasium as gym
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

# Building2Building imports
from building2building.algorithms.online.dqn import QNetwork, dqn_evaluate
from building2building.algorithms.online.ppo import Agent as PPOAgent, ppo_evaluate
from building2building.algorithms.online.baselines import constant_policy
from building2building.core.run_manager import RunManager
from building2building.simulator.wrappers import NormalizeObservation, CustomRescaleAction
import building2building.simulator


def parse_args():
    parser = argparse.ArgumentParser(description="Collect offline datasets for offline RL training")
    
    # Building and environment arguments
    parser.add_argument('--state', '-s', type=str, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, help='County name')
    parser.add_argument('--building-id', '-b', type=str, help='Building ID')
    parser.add_argument('--weather', '-w', type=str, help='EPW Weather file')
    
    # Data collection arguments
    parser.add_argument('--output-path', '-o', type=str, required=True,
                        help='Output path for the dataset (JSON or NPZ format)')
    parser.add_argument('--data-source', type=str, default="mixed",
                        choices=["baseline", "trained_policy", "mixed", "random"],
                        help='Source of data to collect')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes to collect')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    
    # Environment configuration
    parser.add_argument('--reward-type', type=str, default="base",
                        choices=["barrier", "base"], help='Reward type')
    parser.add_argument('--energy-weight', type=float, default=1.0,
                        help='Energy weight for base reward function')
    
    # Policy-specific arguments
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model (required for trained_policy)')
    parser.add_argument('--policy-type', type=str, default="ppo",
                        choices=["dqn", "ppo"], help='Type of trained policy')
    
    # Baseline policy arguments  
    parser.add_argument('--heating-setpoint', type=float, default=21.0,
                        help='Heating setpoint for baseline policy')
    parser.add_argument('--cooling-setpoint', type=float, default=24.0,
                        help='Cooling setpoint for baseline policy')
    
    # Mixed dataset arguments
    parser.add_argument('--baseline-ratio', type=float, default=0.5,
                        help='Ratio of baseline episodes in mixed dataset')
    parser.add_argument('--noise-level', type=float, default=0.1,
                        help='Noise level for action perturbation in mixed dataset')
    
    return parser.parse_args()


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
            action = constant_policy(obs, heating_setpoint, cooling_setpoint, 
                                   normalize=True, action_space=env.env.env.action_space)
            
            # Store current observation and action
            data['observations'].append(obs.copy())
            data['actions'].append(action.copy())
            
            # Take step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store next observation and outcomes
            data['next_observations'].append(next_obs.copy())  
            data['rewards'].append(reward)
            data['terminals'].append(done)
            data['timeouts'].append(truncated)  # D4RL treats this as timeout flag
            
            obs = next_obs
    
    return data


def save_dataset(data: Dict[str, List], output_path: str, logger: logging.Logger):
    """Save dataset to file."""
    
    # Convert lists to numpy arrays
    dataset = {}
    for key, values in data.items():
        dataset[key] = np.array(values)
    
    # Log dataset statistics
    n_transitions = len(dataset['observations'])
    total_reward = np.sum(dataset['rewards'])
    mean_reward = np.mean(dataset['rewards'])
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total transitions: {n_transitions}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Mean reward per step: {mean_reward:.4f}")
    logger.info(f"  Observation shape: {dataset['observations'].shape}")
    logger.info(f"  Action shape: {dataset['actions'].shape}")
    
    # Save based on file extension
    if output_path.endswith('.json'):
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, values in dataset.items():
            json_data[key] = values.tolist()
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f)
            
    elif output_path.endswith('.npz'):
        np.savez_compressed(output_path, **dataset)
        
    else:
        raise ValueError(f"Unsupported output format: {output_path}")
    
    logger.info(f"Dataset saved to: {output_path}")


def main():
    args = parse_args()
    
    # Create run manager for logging
    run_manager = RunManager(
        experiment_name="dataset_collection",
        track_wandb=False,
        seed=args.seed,
        tags={
            "data_source": args.data_source,
            "num_episodes": args.num_episodes,
            "state": args.state,
            "county": args.county,
            "building_id": args.building_id
        }
    )
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{args.state}/{args.county}/{args.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{args.state}/{args.county}/{args.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        run_manager.logger.error(f"Building characteristics file not found: {characteristics_path}")
        exit(1)
    
    # Create environment
    env = gym.make(
        "EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=f"data/weather/{args.weather}",
        building_characteristics=building_characteristics,
        reward_type=args.reward_type,
        energy_weight=args.energy_weight,
        run_manager=run_manager
    )
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    
    # Set seed
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # For now, implement baseline collection - can be extended later
    run_manager.logger.info(f"Starting data collection: {args.data_source}")
    
    data = collect_baseline_data(env, args.num_episodes, args.heating_setpoint,
                               args.cooling_setpoint, run_manager.logger)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save dataset
    save_dataset(data, args.output_path, run_manager.logger)
    
    env.close()
    run_manager.finish()


if __name__ == "__main__":
    main() 