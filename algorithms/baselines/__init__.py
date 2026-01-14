from __future__ import annotations

"""
Baselines package.

This replaces the former monolithic `algorithms/baselines.py` module.

Important compatibility notes:
- External code (scripts/tests) imports `algorithms.baselines` and expects:
  - `run_baseline_rollout` to exist
  - `make_env` to be monkeypatchable via `algorithms.baselines.make_env`
"""

from algorithms.utils import make_env, plot_timeseries
from algorithms.baselines.runner import run_baseline_rollout

__all__ = [
    "make_env",
    "plot_timeseries",
    "run_baseline_rollout",
]

