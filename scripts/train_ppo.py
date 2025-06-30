import hydra
from omegaconf import DictConfig
import os
import gymnasium as gym
import torch
import torch.nn as nn
import json
from building2building.algorithms.online.ppo import main, Args
from building2building.core.hydra_manager import HydraManager

# Make sure to import your environment to register it
import building2building.simulator


def create_ppo_args_from_config(cfg: DictConfig, run_manager: HydraManager) -> Args:
    """Convert Hydra configuration to PPO Args object."""
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        run_manager.logger.error(f"Building characteristics file not found: {characteristics_path}")
        raise
    
    # Create PPO arguments from Hydra config
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
        
        # PPO specific parameters
        learning_rate=cfg.ppo.learning_rate,
        num_envs=cfg.ppo.num_envs,
        num_steps=cfg.ppo.num_steps,
        anneal_lr=cfg.ppo.anneal_lr,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        num_minibatches=cfg.ppo.num_minibatches,
        update_epochs=cfg.ppo.update_epochs,
        norm_adv=cfg.ppo.norm_adv,
        clip_coef=cfg.ppo.clip_coef,
        clip_vloss=cfg.ppo.clip_vloss,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        target_kl=cfg.ppo.target_kl,
        eval_frequency=cfg.ppo.eval_frequency,
    )
    
    return args


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for PPO training with Hydra configuration."""
    
    # Ensure we're using the PPO training experiment
    if cfg.name != "ppo_training":
        raise ValueError(f"This script expects ppo_training experiment, got {cfg.name}")
    
    # Create a Hydra-compatible run manager
    run_manager = HydraManager(cfg)
    
    # Convert Hydra config to PPO Args
    args = create_ppo_args_from_config(cfg, run_manager)
    
    # Log the configuration
    run_manager.save_config()
    
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


if __name__ == "__main__":
    main_hydra()
