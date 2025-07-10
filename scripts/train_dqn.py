#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig
from building2building.training import train_dqn

# Import environment to register it
import building2building.simulator


@hydra.main(version_base=None, config_path="../configs", config_name="dqn_training")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for DQN training with Hydra configuration."""
    train_dqn(cfg)


if __name__ == "__main__":
    main_hydra()
