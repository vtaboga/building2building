from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from b2b.baselines.online_trainer import online_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    online_trainer(config=cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()

