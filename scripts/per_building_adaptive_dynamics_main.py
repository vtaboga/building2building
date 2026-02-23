"""
Entry point for the per-building PPO trainer on the adaptive dynamics benchmark.

Intended to be launched as a SLURM array job:

    python -m scripts.per_building_adaptive_dynamics_main \
        split=test split_index=$SLURM_ARRAY_TASK_ID

Each invocation trains a dedicated PPO agent on a single building.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from algorithms.per_building_adaptive_dynamics_trainer import (
    per_building_adaptive_dynamics_trainer,
)


@hydra.main(
    config_path="../configs",
    config_name="per_building_adaptive_dynamics",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    split: str = str(cfg.split)
    split_index: int = int(cfg.split_index)
    per_building_adaptive_dynamics_trainer(
        config=cfg,
        output_dir=output_dir,
        split=split,
        split_index=split_index,
    )


if __name__ == "__main__":
    main()
