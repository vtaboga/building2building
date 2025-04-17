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
    parser.add_argument('--state', type=str, default="AL", help='State code (e.g., AL)')
    parser.add_argument('--county', type=str, default="Pike", help='County name')
    parser.add_argument('--building-id', type=str, default="6014003401548", help='Building ID')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--track', action='store_true', help='Track with wandb')
    parser.add_argument('--wandb-project', type=str, default="building2building", help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default="pierre-luc-bacon-mila-org", help='W&B entity')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='Total timesteps for training')
    
    return parser.parse_args()

def make_env(env_id, path_to_building, path_to_weather, building_characteristics, run_manager=None):
    def thunk():
        env = gym.make(env_id, 
                      path_to_building=path_to_building, 
                      path_to_weather=path_to_weather, 
                      building_characteristics=building_characteristics,
                      run_manager=run_manager)
        return env
    return thunk

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
        path_to_weather=f"data/weather/USA_{cmd_args.state}_Albertville.Muni.AP.720376_TMYx.2004-2018.epw",
        building_characteristics=building_characteristics,
        track=cmd_args.track,
        wandb_project_name=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        results_dir=run_manager.run_dir,  # Use run manager's directory
        seed=cmd_args.seed,
        total_timesteps=cmd_args.total_timesteps,
        save_model=True  # Always save the model
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