from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from b2b.baselines.multizones_trainer import multizones_trainer


def run_multizones_training(config: dict[str, Any], output_dir: Path) -> None:
    cfg = OmegaConf.create(config)
    multizones_trainer(config=cfg, output_dir=output_dir)
