"""
Training script for offline RL algorithms using Hydra configuration.
"""

import os
import json
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

# Remove the problematic import for now
# from building2building.algorithms.offline.offline_rl import main, OfflineRLArgs
import building2building.simulator


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for offline RL training with Hydra configuration."""
    
    # Ensure we're using the offline RL training experiment
    if cfg.name != "train_offline_rl":
        raise ValueError(f"This script expects train_offline_rl experiment, got {cfg.name}")
    
    # Get Hydra's output directory and setup logging
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger(__name__)
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        logger.error(f"Building characteristics file not found: {characteristics_path}")
        raise
    
    # Validate dataset exists
    dataset_path = cfg.offline_rl.dataset_path
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Log configuration for now (placeholder for actual training)
    logger.info(f"Starting {cfg.offline_rl.algorithm.upper()} offline RL training...")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Building: {cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}")
    logger.info(f"Algorithm: {cfg.offline_rl.algorithm}")
    logger.info(f"Epochs: {cfg.offline_rl.epoch}")
    logger.info(f"Batch size: {cfg.offline_rl.batch_size}")
    logger.info(f"Learning rate: {cfg.offline_rl.lr}")
    
    # TODO: Implement actual offline RL training once RunManager dependency is resolved
    logger.info("Offline RL training configuration validated successfully!")
    logger.info("Note: Actual training implementation requires resolving RunManager dependency")


if __name__ == "__main__":
    main_hydra() 