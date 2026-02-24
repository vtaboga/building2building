#!/usr/bin/env python3
"""Train an SB3 agent on a single multizones_reference_buildings building.

The building is selected by (building_type, split, index) where *index* is
the 0-based position in the pre-generated split ID list.

Usage::

    PYTHONPATH=. python scripts/train_multizones.py

    # Train PPO on the 3rd building of the OfficeMedium train split
    PYTHONPATH=. python scripts/train_multizones.py \\
        multizones.building_type=OfficeMedium multizones.split=train multizones.index=2

    # Use SAC instead of PPO
    PYTHONPATH=. python scripts/train_multizones.py policy=sac

    # Short run for debugging
    PYTHONPATH=. python scripts/train_multizones.py \\
        training.total_timesteps=10000 env.max_steps=960
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from b2b.baselines.multizones_trainer import multizones_trainer


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_multizones",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    multizones_trainer(config=cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
