import os
import gymnasium as gym
import torch
import torch.nn as nn
import json
import argparse
from src.algorithms.dqn import main, Args
from src.core.run_manager import RunManager

# Make sure to import your environment to register it
import src.simulator

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN algorithm on EnergyPlus environment")
    # Building and environment arguments
    parser.add_argument('--state', '-s', type=str, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, help='County name')
    parser.add_argument('--building-id', '-b', type=str, help='Building ID')
    parser.add_argument('--weather', '-w', type=str, help='EPW Weather file')
    parser.add_argument('--weather_validation', type=str, default=None, help='EPW Weather file for validation (optional)')
    
    # General training arguments
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--track', action='store_true', help='Track with wandb')
    parser.add_argument('--wandb-project', type=str, default="building2building", help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default="pierre-luc-bacon-mila-org", help='W&B entity')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='Total timesteps for training')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save the trained model')
    parser.add_argument('--results-dir', type=str, default=None, help='Base directory for storing results')
    
    # DQN specific arguments
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--reward-type', type=str, default="base", choices=["barrier", "base"], help='Reward type')
    parser.add_argument('--energy-weight', type=float, default=1.0, help='Energy weight for base reward function')
    parser.add_argument('--tau', type=float, default=1.0, help='Target network update rate')
    parser.add_argument('--target-network-frequency', type=int, default=500, help='Target network update frequency')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--start-e', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--end-e', type=float, default=0.05, help='Final epsilon for exploration')
    parser.add_argument('--exploration-fraction', type=float, default=0.5, 
                        help='Fraction of total timesteps for epsilon decay')
    parser.add_argument('--learning-starts', type=int, default=10000, help='Timesteps before learning starts')
    parser.add_argument('--train-frequency', type=int, default=10, help='Training frequency')
    parser.add_argument('--eval-frequency', type=int, default=100000, 
                        help='How often (in steps) to evaluate the policy during training')
    
    return parser.parse_args()

if __name__ == "__main__":
    cmd_args = parse_args()
    
    # Create a run manager to handle all logging and result saving
    run_manager = RunManager(
        experiment_name="dqn_training",
        track_wandb=cmd_args.track,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        seed=cmd_args.seed,
        tags={
            "state": cmd_args.state,
            "county": cmd_args.county,
            "building_id": cmd_args.building_id
        }
    )
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{cmd_args.state}/{cmd_args.county}/{cmd_args.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cmd_args.state}/{cmd_args.county}/{cmd_args.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        run_manager.logger.error(f"Building characteristics file not found: {characteristics_path}")
        exit(1)
    
    # Create DQN arguments
    args = Args(
        env_id="EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=f"data/weather/{cmd_args.weather}",
        weather_validation=f"data/weather/{cmd_args.weather_validation}" if cmd_args.weather_validation else None,
        building_characteristics=building_characteristics,
        reward_type=cmd_args.reward_type,
        energy_weight=cmd_args.energy_weight,
        
        # Transfer command line arguments to DQN Args
        track=cmd_args.track,
        wandb_project_name=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        results_dir=cmd_args.results_dir or run_manager.run_dir,
        seed=cmd_args.seed,
        total_timesteps=cmd_args.total_timesteps,
        save_model=cmd_args.save_model,
        
        # DQN specific parameters
        learning_rate=cmd_args.learning_rate,
        num_envs=cmd_args.num_envs,
        buffer_size=cmd_args.buffer_size,
        gamma=cmd_args.gamma,
        tau=cmd_args.tau,
        target_network_frequency=cmd_args.target_network_frequency,
        batch_size=cmd_args.batch_size,
        start_e=cmd_args.start_e,
        end_e=cmd_args.end_e,
        exploration_fraction=cmd_args.exploration_fraction,
        learning_starts=cmd_args.learning_starts,
        train_frequency=cmd_args.train_frequency,
        eval_frequency=cmd_args.eval_frequency,
    )
    
    # Log the configuration
    run_manager.save_config(vars(args))
    
    # Run DQN training
    run_manager.logger.info("Starting DQN training...")
    try:
        main(args, run_manager)
        run_manager.logger.info("DQN training completed successfully")
    except Exception as e:
        run_manager.logger.error(f"Error during DQN training: {e}")
        raise
    finally:
        # Ensure we finalize the run even if there's an error
        run_manager.finish() 