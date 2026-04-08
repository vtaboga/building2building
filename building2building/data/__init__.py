"""Unified dataset access for Building2Building.

Provides download, caching, and metadata queries for the unified
HuggingFace dataset (``vtaboga/building2building_dataset``).
"""

from building2building.data.download import (
    ALL_BUILDING_TYPES,
    BuildingType,
    download_building_type,
    download_metadata,
    download_splits,
    get_building_path,
)
from building2building.data.registry import (
    BuildingInfo,
    BuildingRegistry,
    get_registry,
)

__all__ = [
    "ALL_BUILDING_TYPES",
    "BuildingInfo",
    "BuildingRegistry",
    "BuildingType",
    "download_building_type",
    "download_metadata",
    "download_splits",
    "get_building_path",
    "get_registry",
]
