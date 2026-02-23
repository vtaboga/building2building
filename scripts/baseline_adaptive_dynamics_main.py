"""
Main script for training a baseline PPO policy on the adaptive dynamics benchmark.

Same as parameterized_adaptive_dynamics_main but WITHOUT building parameter augmentation.
The agent does not see building parameters — buildings are still resampled on each reset.
"""

from pathlib import Path

import hydra

from algorithms.parameterized_adaptive_dynamics_trainer import (
    parameterized_adaptive_dynamics_trainer,
)


@hydra.main(config_path="../configs", config_name="baseline_adaptive_dynamics")
def main(cfg):
    output_dir = Path.cwd()
    parameterized_adaptive_dynamics_trainer(config=cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
