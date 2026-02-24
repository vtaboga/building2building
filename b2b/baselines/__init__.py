from __future__ import annotations

"""
Baselines package.

Implements baseline RL policies and classic rule-based controllers.
"""

from b2b.make_env import make_env
from b2b.baselines.policies import make_policy_from_config
from b2b.baselines.utils import plot_timeseries
from b2b.baselines.runner import run_baseline_rollout

# Package exports.
__all__ = [
    "make_env",
    "make_policy_from_config",
    "plot_timeseries",
    "run_baseline_rollout",
]

