import hydra
import wandb
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from typing import Dict, Any, cast
import json
import shutil
import os

from building2building.algorithms.online.ppo import main

# Make sure to import your environment to register it
import building2building.simulator


@hydra.main(version_base=None, config_path="../configs", config_name="ppo_training")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for PPO training with Hydra configuration."""
    
    # Get Hydra's output directory and setup logging
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger(__name__)
    
    # Remove unused subdirectories that Hydra might create
    unused_dirs = ['models', 'data', 'eplus_output']
    for unused_dir in unused_dirs:
        unused_path = output_dir / unused_dir
        if unused_path.exists():
            shutil.rmtree(unused_path)
            logger.info(f"Removed unused directory: {unused_path}")
    
    # Initialize W&B following best practices
    if cfg.get('track', False):
        # Convert config to proper dict for wandb
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            config=cast(Dict[str, Any], config_dict) if isinstance(config_dict, dict) else {},
            name=cfg.get('name', 'ppo_training'),
            tags=cfg.get('tags', [])
        )
        logger.info(f"W&B initialized: {wandb_run.url}")
    
    # Load building characteristics and set up paths
    building_path = f"data/processed_buildings/{cfg.state}/{cfg.county}/{cfg.building_id}.epJSON"
    weather_path = f"data/weather/{cfg.weather}"
    weather_validation_path = f"data/weather/{cfg.weather_validation}" if cfg.get('weather_validation') else None
    characteristics_path = f"data/processed_buildings/{cfg.state}/{cfg.county}/{cfg.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        logger.error(f"Building characteristics file not found: {characteristics_path}")
        raise
    
    # Run PPO training
    logger.info("Starting PPO training...")
    logger.info(f"Results will be saved to: {output_dir}")
    try:
        main(
            cfg=cfg,
            building_path=building_path,
            weather_path=weather_path,
            weather_validation_path=weather_validation_path,
            building_characteristics=building_characteristics,
            results_dir=str(output_dir)
        )
        logger.info("PPO training completed successfully")
    except Exception as e:
        logger.error(f"Error during PPO training: {e}")
        raise
    
    # Hydra automatically saves config to .hydra/config.yaml
    logger.info(f"Config automatically saved to: {output_dir}/.hydra/config.yaml")
    logger.info(f"Final results structure:")
    logger.info(f"  - EnergyPlus outputs: {output_dir}/eplus_outputs/")
    logger.info(f"  - Test results: {output_dir}/test_results/")
    logger.info(f"  - Model: {output_dir}/model.pt")
    logger.info(f"  - Training log: {output_dir}/train_ppo.log")
    logger.info(f"  - TensorBoard logs: {output_dir}/logs/")


if __name__ == "__main__":
    main_hydra()
