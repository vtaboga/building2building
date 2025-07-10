#!/usr/bin/env python3

import hydra
import sys
from omegaconf import DictConfig, OmegaConf
from building2building.algorithms.online.policy_trainer import get_training_function

# Import environment to register it
import building2building.simulator


@hydra.main(version_base=None, config_path="../configs", config_name="online_training")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for online training with Hydra configuration."""
    
    # Determine algorithm from policies config or direct algorithm config
    algorithm = getattr(cfg, 'policies', None)
    if algorithm is None:
        print("Algoritm is None, fallback to algorithm config")
        # Fallback: check for algorithm-specific configs
        if hasattr(cfg, 'ppo') and cfg.ppo is not None:
            algorithm = 'ppo'
        elif hasattr(cfg, 'dqn') and cfg.dqn is not None:
            algorithm = 'dqn'
    
    if algorithm not in ['ppo', 'dqn']:
        print("Error: Invalid or missing algorithm. Please specify policies=ppo or policies=dqn")
        print("Example: python scripts/train_online.py policies=ppo")
        sys.exit(1)
    
    print(f"Starting {algorithm.upper()} training...")
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Get and run the appropriate training function
    try:
        train_func = get_training_function(algorithm)
        train_func(cfg)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_hydra()
