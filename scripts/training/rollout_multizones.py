#!/usr/bin/env python3
"""Run baseline rollouts on multizones_reference_buildings.

Thin Hydra entry point.  All logic lives in
``b2b.benchmark.rollout_multizones``.

Usage::

    python scripts/rollout_multizones.py

    # Override building types and count
    python scripts/rollout_multizones.py \\
        'multizones.types=[OfficeSmall,Warehouse]' multizones.n_per_type=3

    # Different policy
    python scripts/rollout_multizones.py policy=unitary_g36

    # Shorter episodes
    python scripts/rollout_multizones.py env.max_steps=2016
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None, config_path="../configs", config_name="rollout_multizones"
)
def main(cfg: DictConfig) -> int:
    from building2building.benchmark.rollout_multizones import run_multizones_rollout

    cfg_dict_any = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = cfg_dict_any if isinstance(cfg_dict_any, dict) else {}
    records = run_multizones_rollout(cfg_dict, output_dir=Path.cwd())
    n_fail = sum(1 for r in records if not r.success)
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
