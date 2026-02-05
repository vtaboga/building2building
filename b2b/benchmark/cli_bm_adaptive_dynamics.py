from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from b2b.benchmark.experiments.bm_adaptive_dynamics import run_bm_adaptive_dynamics


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> int:
    return int(run_bm_adaptive_dynamics(cfg, output_dir=Path.cwd()))


if __name__ == "__main__":
    raise SystemExit(main())
