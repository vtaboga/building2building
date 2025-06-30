import hydra
import wandb
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from typing import Dict, Any, cast
import json

from building2building.algorithms.online.ppo import main, Args

# Make sure to import your environment to register it
import building2building.simulator


def create_ppo_args_from_config(cfg: DictConfig, output_dir: Path) -> Args:
    """Convert Hydra configuration to PPO Args object."""
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        logging.error(f"Building characteristics file not found: {characteristics_path}")
        raise
    
    # Handle weather_validation parameter (Args class expects str, not Optional[str])
    weather_validation = f"data/weather/{cfg.building.weather_validation}" if cfg.building.get('weather_validation') else cfg.building.weather
    
    # Create PPO arguments from Hydra config
    args = Args(
        # Environment settings
        env_id="EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=f"data/weather/{cfg.building.weather}",
        weather_validation=weather_validation,
        building_characteristics=building_characteristics,
        reward_type=cfg.env.reward_type,
        energy_weight=cfg.env.energy_weight,
        
        # Tracking settings
        track=cfg.track,
        wandb_project_name=cfg.wandb.project,
        wandb_entity=cfg.wandb.entity,
        results_dir=str(output_dir),
        
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
    
    # Get Hydra's output directory and setup logging
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger(__name__)
    
    # Create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'eplus_output').mkdir(exist_ok=True)
    
    # Initialize W&B following best practices
    wandb_run = None
    if cfg.get('track', False):
        # Convert config to proper dict for wandb
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=cast(Dict[str, Any], config_dict) if isinstance(config_dict, dict) else {},
            name=cfg.get('name', 'ppo_training'),
            tags=cfg.wandb.get('tags', [])
        )
        logger.info(f"W&B initialized: {wandb_run.url}")
    
    # Convert Hydra config to PPO Args
    args = create_ppo_args_from_config(cfg, output_dir)
    
    # Run PPO training
    logger.info("Starting PPO training...")
    try:
        # Pass None as run_manager since we're using Hydra directly
        main(args, run_manager=None)
        logger.info("PPO training completed successfully")
    except Exception as e:
        logger.error(f"Error during PPO training: {e}")
        raise
    
    # Hydra automatically saves config to .hydra/config.yaml
    logger.info(f"Config automatically saved to: {output_dir}/.hydra/config.yaml")


if __name__ == "__main__":
    main_hydra()
