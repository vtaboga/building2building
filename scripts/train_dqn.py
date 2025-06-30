import hydra
from omegaconf import DictConfig
import os
import gymnasium as gym
import torch
import torch.nn as nn
import json
from building2building.algorithms.online.dqn import main, Args
from building2building.core.hydra_manager import HydraManager

# Make sure to import your environment to register it
import building2building.simulator


def create_dqn_args_from_config(cfg: DictConfig, run_manager: HydraManager) -> Args:
    """Convert Hydra configuration to DQN Args object."""
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        run_manager.logger.error(f"Building characteristics file not found: {characteristics_path}")
        raise
    
    # Create DQN arguments from Hydra config
    args = Args(
        # Environment settings
        env_id="EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=f"data/weather/{cfg.building.weather}",
        weather_validation=f"data/weather/{cfg.building.weather_validation}" if cfg.building.get('weather_validation') else None,
        building_characteristics=building_characteristics,
        reward_type=cfg.env.reward_type,
        energy_weight=cfg.env.energy_weight,
        
        # Tracking settings
        track=cfg.track,
        wandb_project_name=cfg.wandb.project,
        wandb_entity=cfg.wandb.entity,
        results_dir=run_manager.run_dir,
        
        # General training settings
        seed=cfg.seed,
        total_timesteps=cfg.training.total_timesteps,
        save_model=cfg.training.save_model,
        
        # DQN specific parameters
        learning_rate=cfg.dqn.learning_rate,
        num_envs=cfg.dqn.num_envs,
        buffer_size=cfg.dqn.buffer_size,
        gamma=cfg.dqn.gamma,
        tau=cfg.dqn.tau,
        target_network_frequency=cfg.dqn.target_network_frequency,
        batch_size=cfg.dqn.batch_size,
        start_e=cfg.dqn.start_e,
        end_e=cfg.dqn.end_e,
        exploration_fraction=cfg.dqn.exploration_fraction,
        learning_starts=cfg.dqn.learning_starts,
        train_frequency=cfg.dqn.train_frequency,
        eval_frequency=cfg.dqn.eval_frequency,
    )
    
    return args


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for DQN training with Hydra configuration."""
    
    # Ensure we're using the DQN training experiment
    if cfg.name != "dqn_training":
        raise ValueError(f"This script expects dqn_training experiment, got {cfg.name}")
    
    # Create a Hydra-compatible run manager
    run_manager = HydraManager(cfg)
    
    # Convert Hydra config to DQN Args
    args = create_dqn_args_from_config(cfg, run_manager)
    
    # Log the configuration
    run_manager.save_config()
    
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


if __name__ == "__main__":
    main_hydra()
