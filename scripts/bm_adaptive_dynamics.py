from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Ensure Hydra outputs go under the repo's `outputs/` even if invoked elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("B2B_REPO_ROOT", str(_REPO_ROOT))


@hydra.main(version_base=None, config_path="../configs", config_name="bm_adaptive_dynamics")
def main(cfg: DictConfig) -> int:
    from b2b.benchmark.experiments.bm_adaptive_dynamics import (  # noqa: WPS433
        run_bm_adaptive_dynamics,
    )

    return int(run_bm_adaptive_dynamics(cfg, output_dir=Path.cwd()))


if __name__ == "__main__":
    # Allow running as `python scripts/bm_adaptive_dynamics.py` without installing the package.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    raise SystemExit(main())

