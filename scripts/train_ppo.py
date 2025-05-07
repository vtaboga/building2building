import os
import gymnasium as gym
import torch
import torch.nn as nn
import json
import argparse
from src.algorithms.ppo import main, Args
from src.core.run_manager import RunManager

# Make sure to import your environment to register it
import src.simulator

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO algorithm on EnergyPlus environment")
    # Building and environment arguments
    parser.add_argument('--state', '-s', type=str, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, help='County name')
    parser.add_argument('--building-id', '-b', type=str, help='Building ID')
    parser.add_argument('--weather', '-w', type=str, help='EPW Weather file')
    
    # General training arguments
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--track', action='store_true', help='Track with wandb')
    parser.add_argument('--wandb-project', type=str, default="building2building", help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default="pierre-luc-bacon-mila-org", help='W&B entity')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='Total timesteps for training')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save the trained model')
    parser.add_argument('--results-dir', type=str, default=None, help='Base directory for storing results')
    
    # PPO specific arguments
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=2048, help='Steps per environment per rollout')
    parser.add_argument('--anneal-lr', action='store_true', default=True, help='Anneal learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--reward-type', type=str, default="base", choices=["barrier", "base"], help='Reward type')
    parser.add_argument('--energy-weight', type=float, default=1.0, help='Energy weight for base reward function')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--num-minibatches', type=int, default=32, help='Number of minibatches')
    parser.add_argument('--update-epochs', type=int, default=10, help='Number of update epochs')
    parser.add_argument('--norm-adv', action='store_true', default=True, help='Normalize advantages')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='PPO clipping coefficient')
    parser.add_argument('--clip-vloss', action='store_true', default=True, help='Clip value loss')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--target-kl', type=float, default=None, help='Target KL divergence')
    parser.add_argument('--eval-frequency', type=int, default=100000, 
                        help='How often (in steps) to evaluate the policy during training')
    
    return parser.parse_args()

if __name__ == "__main__":
    cmd_args = parse_args()
    
    # Create a run manager to handle all logging and result saving
    run_manager = RunManager(
        experiment_name="ppo_training",
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
    
    # Create PPO arguments
    args = Args(
        env_id="EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=f"data/weather/{cmd_args.weather}",
        building_characteristics=building_characteristics,
        reward_type=cmd_args.reward_type,
        
        # Transfer command line arguments to PPO Args
        track=cmd_args.track,
        wandb_project_name=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        results_dir=cmd_args.results_dir or run_manager.run_dir,
        seed=cmd_args.seed,
        total_timesteps=cmd_args.total_timesteps,
        save_model=cmd_args.save_model,
        
        # PPO specific parameters
        learning_rate=cmd_args.learning_rate,
        num_envs=cmd_args.num_envs,
        num_steps=cmd_args.num_steps,
        anneal_lr=cmd_args.anneal_lr,
        gamma=cmd_args.gamma,
        gae_lambda=cmd_args.gae_lambda,
        num_minibatches=cmd_args.num_minibatches,
        update_epochs=cmd_args.update_epochs,
        norm_adv=cmd_args.norm_adv,
        clip_coef=cmd_args.clip_coef,
        clip_vloss=cmd_args.clip_vloss,
        ent_coef=cmd_args.ent_coef,
        vf_coef=cmd_args.vf_coef,
        max_grad_norm=cmd_args.max_grad_norm,
        target_kl=cmd_args.target_kl,
        eval_frequency=cmd_args.eval_frequency,
    )
    
    # Log the configuration
    run_manager.save_config(vars(args))
    
    # Run PPO training
    run_manager.logger.info("Starting PPO training...")
    try:
        main(args, run_manager)
        run_manager.logger.info("PPO training completed successfully")
    except Exception as e:
        run_manager.logger.error(f"Error during PPO training: {e}")
        raise
    finally:
        # Ensure we finalize the run even if there's an error
        run_manager.finish() 