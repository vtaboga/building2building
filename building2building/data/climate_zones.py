"""Climate-zone helpers for the Building2Building dataset.

This module is intentionally minimal: the ``climate_zone`` column in the
unified ``metadata.parquet`` is the single source of truth. City->CZ mappings
and weather-file parsing live only at the data-building layer
(:mod:`scripts.processing.merge_datasets`) and are applied once before upload
to HuggingFace.

Library consumers should use:

* :func:`building2building.api.list_buildings_by_climate_zone`
* :func:`building2building.api.get_climate_zone`

both of which read the column directly via :class:`BuildingRegistry`.
"""

from __future__ import annotations

__all__ = [
    "ClimateZoneUnavailableError",
    "TYPES_WITHOUT_CLIMATE_ZONE",
]


class ClimateZoneUnavailableError(ValueError):
    """Raised when a building type has no ASHRAE climate-zone assignment.

    In the current dataset only :data:`TYPES_WITHOUT_CLIMATE_ZONE` members
    trigger this error: SingleFamilyHouse uses per-building Canadian weather
    files that are not mapped to ASHRAE climate zones.
    """


TYPES_WITHOUT_CLIMATE_ZONE: frozenset[str] = frozenset({"SingleFamilyHouse"})
