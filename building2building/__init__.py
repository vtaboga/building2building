"""Building2Building: A benchmark suite for building control with RL."""

try:
    from building2building.env import setup_energyplus_path

    setup_energyplus_path()
except ModuleNotFoundError:
    pass

from building2building.api import (
    list_building_types,
    list_buildings,
    make_env,
    make_multizones_env,
    make_single_zone_env,
    new_make_env,
)
from building2building.scoring import compute_normalized_score
from building2building.types import (
    BarrierRewardConfig,
    BaseRewardConfig,
    DeadbandRewardConfig,
)

import building2building.benchmarks as benchmarks  # noqa: F401

from building2building.envs.registration import register_all as _register_envs

_register_envs()

__all__ = [
    "benchmarks",
    "compute_normalized_score",
    "list_building_types",
    "list_buildings",
    "make_env",
    "make_multizones_env",
    "make_single_zone_env",
    "new_make_env",
    "BarrierRewardConfig",
    "BaseRewardConfig",
    "DeadbandRewardConfig",
]
