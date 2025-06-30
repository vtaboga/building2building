import hydra
import wandb
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from typing import Dict, Any, cast
import json

from building2building.algorithms.online.dqn import main, Args

# Make sure to import your environment to register it
import building2building.simulator


def create_dqn_args_from_config(cfg: DictConfig, output_dir: Path) -> Args:
    """Convert Hydra configuration to DQN Args object."""
    
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
    
    # Create DQN arguments from Hydra config
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
            name=cfg.get('name', 'dqn_training'),
            tags=cfg.wandb.get('tags', [])
        )
        logger.info(f"W&B initialized: {wandb_run.url}")
    
    # Convert Hydra config to DQN Args
    args = create_dqn_args_from_config(cfg, output_dir)
    
    # Run DQN training
    logger.info("Starting DQN training...")
    try:
        # Pass None as run_manager since we're using Hydra directly
        main(args, run_manager=None)
        logger.info("DQN training completed successfully")
    except Exception as e:
        logger.error(f"Error during DQN training: {e}")
        raise
    
    # Hydra automatically saves config to .hydra/config.yaml
    logger.info(f"Config automatically saved to: {output_dir}/.hydra/config.yaml")


if __name__ == "__main__":
    main_hydra()
