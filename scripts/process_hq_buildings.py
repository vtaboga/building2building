from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="hq_buildings")
def main(cfg: DictConfig) -> int:
    from b2b.benchmark.processing.hq_buildings import (  # noqa: WPS433
        run_hq_buildings_processing,
    )

    _ = run_hq_buildings_processing(cfg, output_dir=Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

