#!/usr/bin/env python3

import hydra
import sys
from omegaconf import DictConfig, OmegaConf
from building2building.algorithms.online.policy_trainer import get_training_function
import logging

# Import environment to register it
import building2building.simulator

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="online_training")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for online training with Hydra configuration."""
    
    # Determine policy type from algorithm config
    if hasattr(cfg, 'ppo') and cfg.ppo is not None:
        algorithm = 'ppo'
    elif hasattr(cfg, 'dqn') and cfg.dqn is not None:
        algorithm = 'dqn'
    else:
        raise ValueError("Invalid or missing algorithm. Please specify policies=ppo or policies=dqn")

    logging.info(f"Starting {algorithm.upper()} training")
    
    # Get and run the appropriate training function
    try:
        train_func = get_training_function(algorithm)
        train_func(cfg)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_hydra()
