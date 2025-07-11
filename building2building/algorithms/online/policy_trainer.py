"""
Training module for building2building algorithms.
This module contains the training logic that can be imported and used by scripts.
"""

import hydra
import wandb
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from typing import Dict, Any, cast, Callable
import json
import hashlib

from building2building.algorithms.online.ppo import main as ppo_main
from building2building.algorithms.online.dqn import main as dqn_main

# Make sure to import your environment to register it
import building2building.simulator

# Algorithm registry
ALGORITHMS = {
    "ppo": ppo_main,
    "dqn": dqn_main,
}


def _setup_wandb(cfg: DictConfig, logger: logging.Logger) -> Any:
    """Initialize W&B tracking if enabled."""
    if not cfg.get('track', False):
        return None
    
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        
        # Get Hydra's sweep directory name for consistent group naming
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode.name == "MULTIRUN":
            # Use the sweep directory path to create a consistent short hash
            sweep_dir = Path(hydra_cfg.sweep.dir)
            sweep_id = hashlib.md5(str(sweep_dir).encode()).hexdigest()[:8]
        else:
            # Single run mode - use run directory
            run_dir = Path(hydra_cfg.run.dir)
            sweep_id = hashlib.md5(str(run_dir).encode()).hexdigest()[:8]
        
        # Create unique run name with seed
        group_name = f"{config_dict['policy_type']}_bldg_{config_dict['building_id']}_{sweep_id}"
        run_name = f"{config_dict['policy_type']}_bldg_{config_dict['building_id']}_seed{config_dict['seed']}"
        
        wandb_run = wandb.init(
            entity=cfg.get('entity'),
            project=cfg.get('project'),
            config=cast(Dict[str, Any], config_dict) if isinstance(config_dict, dict) else {},
            name=run_name,
            group=group_name,
            tags=cfg.get('tags', []),
            sync_tensorboard=True,
            reinit=True
        )
        logger.info(f"W&B initialized: {wandb_run.url}")
        return wandb_run
    except Exception as e:
        logger.error(f"W&B initialization failed: {e}")
        logger.error("Continuing without W&B tracking...")
        return None


def _load_building_data(cfg: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Load building characteristics and return file paths."""
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
    
    return {
        'building_path': building_path,
        'weather_path': weather_path,
        'weather_validation_path': weather_validation_path,
        'building_characteristics': building_characteristics
    }


def train_algorithm(cfg: DictConfig, algorithm: str, results_dir: str = None) -> None:
    """Train the specified algorithm with the given configuration."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys())}")
    
    # Setup
    output_dir = Path(results_dir) if results_dir else Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger(__name__)
    
    # Initialize W&B (uses Hydra's directory naming for consistent grouping)
    wandb_run = _setup_wandb(cfg, logger)
    
    # Load building data
    building_data = _load_building_data(cfg, logger)
    
    # Run training
    logger.info(f"Starting {algorithm.upper()} training...")
    logger.info(f"Results will be saved to: {output_dir}")
    
    try:
        ALGORITHMS[algorithm](
            cfg=cfg,
            results_dir=str(output_dir),
            wandb_run=wandb_run,
            **building_data
        )
        logger.info(f"{algorithm.upper()} training completed successfully")
    except Exception as e:
        logger.error(f"Error during {algorithm.upper()} training: {e}")
        raise
    
    logger.info(f"Config automatically saved to: {output_dir}/.hydra/config.yaml")


def train_ppo(cfg: DictConfig, results_dir: str = None) -> None:
    """Train PPO algorithm with the given configuration."""
    train_algorithm(cfg, "ppo", results_dir)


def train_dqn(cfg: DictConfig, results_dir: str = None) -> None:
    """Train DQN algorithm with the given configuration."""
    train_algorithm(cfg, "dqn", results_dir)


def get_training_function(algorithm: str) -> Callable:
    """Get the appropriate training function for the given algorithm."""
    if algorithm == "ppo":
        return train_ppo
    elif algorithm == "dqn":
        return train_dqn
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
