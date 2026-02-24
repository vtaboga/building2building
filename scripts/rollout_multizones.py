#!/usr/bin/env python3
"""Run baseline rollouts on multizones_reference_buildings.

Thin Hydra entry point.  All logic lives in
``b2b.benchmark.rollout_multizones``.

Usage::

    PYTHONPATH=. python scripts/rollout_multizones.py

    # Override building types and count
    PYTHONPATH=. python scripts/rollout_multizones.py \\
        'multizones.types=[OfficeSmall,Warehouse]' multizones.n_per_type=3

    # Different policy
    PYTHONPATH=. python scripts/rollout_multizones.py policy=fan_coil_constant

    # Shorter episodes
    PYTHONPATH=. python scripts/rollout_multizones.py env.max_steps=672
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None, config_path="../configs", config_name="rollout_multizones"
)
def main(cfg: DictConfig) -> int:
    from b2b.benchmark.rollout_multizones import run_multizones_rollout

    records = run_multizones_rollout(cfg, output_dir=Path.cwd())
    n_fail = sum(1 for r in records if not r.success)
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
