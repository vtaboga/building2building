"""
Training script for offline RL algorithms on Building2Building environment.
Supports CQL, IQL, and TD3+BC from OfflineRL-Kit.
"""

import os
import json
import argparse
from building2building.algorithms.offline_rl import main, OfflineRLArgs
from building2building.core.run_manager import RunManager

# Make sure to import your environment to register it
import src.simulator


def parse_args():
    parser = argparse.ArgumentParser(description="Train offline RL algorithms on EnergyPlus environment")
    
    # Building and environment arguments
    parser.add_argument('--state', '-s', type=str, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, help='County name')
    parser.add_argument('--building-id', '-b', type=str, help='Building ID')
    parser.add_argument('--weather', '-w', type=str, help='EPW Weather file')
    parser.add_argument('--weather_validation', type=str, default=None, help='EPW Weather file for validation (optional)')
    
    # Dataset arguments
    parser.add_argument('--dataset-path', '-d', type=str, required=True, 
                        help='Path to offline dataset (JSON or NPZ format)')
    
    # General training arguments
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--track', action='store_true', help='Track with wandb')
    parser.add_argument('--wandb-project', type=str, default="building2building-offline", help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default="pierre-luc-bacon-mila-org", help='W&B entity')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save the trained model')
    parser.add_argument('--results-dir', type=str, default=None, help='Base directory for storing results')
    
    # Algorithm arguments
    parser.add_argument('--algorithm', type=str, default="cql", 
                        choices=["cql", "iql", "td3bc"], help='Offline RL algorithm to use')
    
    # Training configuration
    parser.add_argument('--epoch', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--step-per-epoch', type=int, default=1000, help='Number of training steps per epoch')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    # Environment arguments
    parser.add_argument('--reward-type', type=str, default="base", 
                        choices=["barrier", "base"], help='Reward type')
    parser.add_argument('--energy-weight', type=float, default=1.0, 
                        help='Energy weight for base reward function')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    
    # Algorithm-specific arguments
    # CQL
    parser.add_argument('--cql-weight', type=float, default=1.0, help='CQL regularization weight')
    parser.add_argument('--temperature', type=float, default=1.0, help='CQL temperature parameter')
    parser.add_argument('--with-lagrange', action='store_true', help='Use Lagrange multiplier for CQL')
    parser.add_argument('--lagrange-threshold', type=float, default=10.0, 
                        help='Lagrange threshold for CQL')
    parser.add_argument('--cql-alpha-lr', type=float, default=3e-4, 
                        help='Learning rate for CQL alpha parameter')
    
    # IQL
    parser.add_argument('--beta', type=float, default=3.0, help='IQL beta parameter')
    parser.add_argument('--iql-tau', type=float, default=0.7, help='IQL tau parameter')
    parser.add_argument('--max-target-backup', action='store_true', help='Use max target backup for IQL')
    
    # TD3+BC
    parser.add_argument('--policy-noise', type=float, default=0.2, help='Policy noise for TD3+BC')
    parser.add_argument('--noise-clip', type=float, default=0.5, help='Noise clip for TD3+BC')
    parser.add_argument('--policy-freq', type=int, default=2, help='Policy update frequency for TD3+BC')
    parser.add_argument('--alpha', type=float, default=2.5, help='Behavioral cloning weight for TD3+BC')
    
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_args()
    
    # Create a run manager to handle all logging and result saving
    run_manager = RunManager(
        experiment_name=f"offline_rl_{cmd_args.algorithm}",
        track_wandb=cmd_args.track,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        seed=cmd_args.seed,
        tags={
            "algorithm": cmd_args.algorithm,
            "state": cmd_args.state,
            "county": cmd_args.county,
            "building_id": cmd_args.building_id,
            "dataset": os.path.basename(cmd_args.dataset_path)
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
    
    # Validate dataset exists
    if not os.path.exists(cmd_args.dataset_path):
        run_manager.logger.error(f"Dataset file not found: {cmd_args.dataset_path}")
        exit(1)
    
    # Create offline RL arguments
    args = OfflineRLArgs(
        exp_name=f"{cmd_args.algorithm}_offline_rl",
        env_id="EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=f"data/weather/{cmd_args.weather}",
        weather_validation=f"data/weather/{cmd_args.weather_validation}" if cmd_args.weather_validation else None,
        building_characteristics=building_characteristics,
        reward_type=cmd_args.reward_type,
        energy_weight=cmd_args.energy_weight,
        
        # Dataset configuration
        dataset_path=cmd_args.dataset_path,
        
        # Algorithm configuration
        algorithm=cmd_args.algorithm,
        
        # Training configuration
        track=cmd_args.track,
        wandb_project_name=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        results_dir=cmd_args.results_dir or run_manager.run_dir,
        seed=cmd_args.seed,
        save_model=cmd_args.save_model,
        
        epoch=cmd_args.epoch,
        step_per_epoch=cmd_args.step_per_epoch,
        eval_episodes=cmd_args.eval_episodes,
        batch_size=cmd_args.batch_size,
        lr=cmd_args.lr,
        gamma=cmd_args.gamma,
        tau=cmd_args.tau,
        
        # Algorithm-specific parameters
        # CQL
        cql_weight=cmd_args.cql_weight,
        temperature=cmd_args.temperature,
        with_lagrange=cmd_args.with_lagrange,
        lagrange_threshold=cmd_args.lagrange_threshold,
        cql_alpha_lr=cmd_args.cql_alpha_lr,
        
        # IQL
        beta=cmd_args.beta,
        iql_tau=cmd_args.iql_tau,
        max_target_backup=cmd_args.max_target_backup,
        
        # TD3+BC
        policy_noise=cmd_args.policy_noise,
        noise_clip=cmd_args.noise_clip,
        policy_freq=cmd_args.policy_freq,
        alpha=cmd_args.alpha,
    )
    
    # Log the configuration
    run_manager.save_config(vars(args))
    
    # Run offline RL training
    run_manager.logger.info(f"Starting {cmd_args.algorithm.upper()} offline RL training...")
    run_manager.logger.info(f"Dataset: {cmd_args.dataset_path}")
    run_manager.logger.info(f"Building: {cmd_args.state}/{cmd_args.county}/{cmd_args.building_id}")
    
    try:
        main(args, run_manager)
        run_manager.logger.info(f"{cmd_args.algorithm.upper()} training completed successfully")
    except Exception as e:
        run_manager.logger.error(f"Error during {cmd_args.algorithm.upper()} training: {e}")
        raise
    finally:
        # Ensure we finalize the run even if there's an error
        run_manager.finish() 