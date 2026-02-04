"""
Main script for training a parameterized policy across multiple buildings.

This script trains a single policy that can generalize to different buildings
by augmenting observations with building-specific parameters.
"""

import hydra
from pathlib import Path

from algorithms.parameterized_trainer import parameterized_trainer


@hydra.main(config_path="../configs", config_name="parameterized")
def main(cfg):
    output_dir = Path.cwd()
    parameterized_trainer(config=cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()

