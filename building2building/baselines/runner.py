from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from building2building.baselines.common import RolloutPaths
from building2building.benchmark.baseline_rollout import run_baseline_rollout as _run_baseline_rollout

logger = logging.getLogger(__name__)


def _get_make_env():
    """
    Resolve `make_env` dynamically from the `building2building.baselines` package.

    This preserves the existing test pattern where tests monkeypatch
    `building2building.baselines.make_env` and expect it to affect `run_baseline_rollout`.
    """
    import building2building.baselines as baselines_pkg

    return baselines_pkg.make_env


def run_baseline_rollout(cfg: DictConfig, *, run_dir: Path | None = None) -> RolloutPaths:
    """
    Rollout a simulation with a baseline policy.

    This is a compatibility wrapper. The implementation lives in
    `building2building.benchmark.baseline_rollout` so simulation execution code is centralized.
    """
    return _run_baseline_rollout(cfg, make_env=_get_make_env(), run_dir=run_dir)

