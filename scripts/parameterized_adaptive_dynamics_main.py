"""
Main script for training a parameterized policy on the adaptive dynamics benchmark.

This script trains a single policy that can generalize across 900 train buildings
and evaluates on 100 test buildings from the Hydro-Québec dataset.
"""

from pathlib import Path

import hydra

from algorithms.parameterized_adaptive_dynamics_trainer import (
    parameterized_adaptive_dynamics_trainer,
)


@hydra.main(config_path="../configs", config_name="parameterized_adaptive_dynamics")
def main(cfg):
    output_dir = Path.cwd()
    parameterized_adaptive_dynamics_trainer(config=cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
