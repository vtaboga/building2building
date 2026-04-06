from __future__ import annotations

"""
Baselines package.

Implements baseline RL policies and classic rule-based controllers.
"""

from building2building.api import make_env_from_hydra_config as make_env
from building2building.baselines.policies import make_policy_from_config
from building2building.baselines.utils import plot_timeseries
from building2building.baselines.runner import run_baseline_rollout

# Package exports.
__all__ = [
    "make_env",
    "make_policy_from_config",
    "plot_timeseries",
    "run_baseline_rollout",
]

