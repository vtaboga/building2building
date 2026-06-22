"""Building2Building: A benchmark suite for building control with RL."""

try:
    from building2building.env import setup_energyplus_path

    setup_energyplus_path()
except ModuleNotFoundError:
    pass

from building2building.api import (
    ClimateZoneUnavailableError,
    Controller,
    TYPES_WITHOUT_CLIMATE_ZONE,
    Trajectory,
    callable_controller,
    get_climate_zone,
    list_building_types,
    list_buildings,
    list_buildings_by_climate_zone,
    make_env,
    make_env_from_config,
    rollout,
)
from building2building.api.rl_wrappers import wrap_env_for_rl
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
    BuildingConfig,
    Equipment,
    NormalizedDeadbandRewardConfig,
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
    "list_buildings_by_climate_zone",
    "get_climate_zone",
    "ClimateZoneUnavailableError",
    "TYPES_WITHOUT_CLIMATE_ZONE",
    "make_env",
    "make_env_from_config",
    # Rollout / trajectory capture
    "Controller",
    "Trajectory",
    "callable_controller",
    "rollout",
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
    "BuildingConfig",
    "Equipment",
    "NormalizedDeadbandRewardConfig",
    "RewardConfig",
    "TaskConfig",
    # Wrappers
    "AugmentObservationWithBuildingParams",
    "NormalizeObservation",
    "PadObservation",
    "ResampleBuildingOnResetWrapper",
    "wrap_env_for_rl",
]
