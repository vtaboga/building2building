#!/usr/bin/env python3
"""Train an SB3 agent on a single multizones_reference_buildings building.

The building is selected by (building_type, split, index) where *index* is
the 0-based position in the pre-generated split ID list.

Usage::

    python scripts/train_multizones.py

    # Train PPO on the 3rd building of the OfficeMedium train split
    python scripts/train_multizones.py \\
        bldg.building_type=OfficeMedium bldg.split=train bldg.index=2

    # Use SAC instead of PPO
    python scripts/train_multizones.py policy=sac

    # Short run for debugging
    python scripts/train_multizones.py \\
        training.total_timesteps=10000 env.max_steps=2880
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from building2building.training import run_multizones_training


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_multizones",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    cfg_dict_any = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = cfg_dict_any if isinstance(cfg_dict_any, dict) else {}
    run_multizones_training(config=cfg_dict, output_dir=output_dir)


if __name__ == "__main__":
    main()
