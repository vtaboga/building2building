"""Unified building selection and dataset access.

Provides helpers for listing available buildings, selecting by split
index or building ID, random sampling, and metadata-based queries across
both the ``single_zone_houses`` and ``multizones_reference_buildings``
datasets.
"""

from building2building.datasets.access import (
    building_id_from_split_index,
    building_ids_from_split_indices,
    query_building_ids,
    sample_building_ids,
    search_config,
    select_building_ids,
)

__all__ = [
    "building_id_from_split_index",
    "building_ids_from_split_indices",
    "query_building_ids",
    "sample_building_ids",
    "search_config",
    "select_building_ids",
]
