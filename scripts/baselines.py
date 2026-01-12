from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from algorithms.baselines import run_baseline_rollout

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    run_baseline_rollout(cfg)


if __name__ == "__main__":
    main()


