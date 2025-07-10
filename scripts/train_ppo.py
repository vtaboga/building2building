#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig
from building2building.training import train_ppo

# Import environment to register it
import building2building.simulator


@hydra.main(version_base=None, config_path="../configs", config_name="ppo_training")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for PPO training with Hydra configuration."""
    train_ppo(cfg)


if __name__ == "__main__":
    main_hydra()
