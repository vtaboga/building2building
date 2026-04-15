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
    new_make_env,
)
from building2building.morphology import (
    Morphology,
    MorphologyEdge,
    MorphologyNode,
    NodeType,
    build_morphology,
    ALL_NODE_TYPES,
    CALENDAR,
    ENERGY,
    HEATING_ZONE,
    UNCONTROLLED_ZONE,
    UNITARY_ZONE,
    VAV_SUPPLY,
    VAV_ZONE,
    VAV_ZONE_NO_COOLING,
    WEATHER,
)
from building2building.pipeline.actuators import (
    HeatingOnlyZone,
    HeatPump,
    UnitarySystem,
    VAVSystem,
    VAVTerminal,
)
from building2building.scoring import compute_normalized_score
from building2building.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
    PadObservation,
    ResampleBuildingOnResetWrapper,
)
from building2building.types import (
    ActuatorDescription,
    BarrierRewardConfig,
    BaseRewardConfig,
    BuildingConfig,
    DeadbandRewardConfig,
    Equipment,
    RewardConfig,
    TaskConfig,
)

import building2building.benchmarks as benchmarks  # noqa: F401

from building2building.envs.registration import register_all as _register_envs

_register_envs()

__all__ = [
    # Environment creation
    "list_building_types",
    "list_buildings",
    "make_env",
    "new_make_env",
    # Scoring
    "compute_normalized_score",
    # Benchmarks
    "benchmarks",
    # Morphology (structured representation)
    "Morphology",
    "MorphologyEdge",
    "MorphologyNode",
    "NodeType",
    "build_morphology",
    "ALL_NODE_TYPES",
    "CALENDAR",
    "ENERGY",
    "HEATING_ZONE",
    "UNCONTROLLED_ZONE",
    "UNITARY_ZONE",
    "VAV_SUPPLY",
    "VAV_ZONE",
    "VAV_ZONE_NO_COOLING",
    "WEATHER",
    # Equipment types
    "HeatingOnlyZone",
    "HeatPump",
    "UnitarySystem",
    "VAVSystem",
    "VAVTerminal",
    # Type definitions
    "ActuatorDescription",
    "BarrierRewardConfig",
    "BaseRewardConfig",
    "BuildingConfig",
    "DeadbandRewardConfig",
    "Equipment",
    "RewardConfig",
    "TaskConfig",
    # Wrappers
    "AugmentObservationWithBuildingParams",
    "NormalizeObservation",
    "PadObservation",
    "ResampleBuildingOnResetWrapper",
]
