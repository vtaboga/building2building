from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="bm_adaptive_dynamics")
def main(cfg: DictConfig) -> int:
    from building2building.benchmark.experiments.bm_adaptive_dynamics import (  # noqa: WPS433
        run_bm_adaptive_dynamics,
    )

    return int(run_bm_adaptive_dynamics(cfg, output_dir=Path.cwd()))


if __name__ == "__main__":
    raise SystemExit(main())

