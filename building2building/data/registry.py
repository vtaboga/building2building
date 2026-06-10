"""Building registry: metadata queries, building selection, and split management.

Provides :class:`BuildingRegistry` which loads the unified metadata parquet
and ``splits.json`` to support building selection by type, split, and index.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from building2building.data.climate_zones import (
    TYPES_WITHOUT_CLIMATE_ZONE,
    ClimateZoneUnavailableError,
)
from building2building.data.download import (
    ALL_BUILDING_TYPES,
    BuildingType,
    download_metadata,
    download_splits,
    get_building_path,
)

logger = logging.getLogger(__name__)

SplitName = Literal["train", "test", "test_small"]

# Canonical ``test_small`` definition (mirrored by
# ``baselines/scripts/make_test_small_split.py``): a curated subset of the
# ``test`` split with TEST_SMALL_SIZE buildings per type — commercial types
# take the first (sorted) test building per ASHRAE climate zone 1..8;
# SingleFamilyHouse takes TEST_SMALL_SIZE seeded-random test buildings.
TEST_SMALL_SIZE = 8
TEST_SMALL_SFH_SEED = 0
_TEST_SMALL_CLIMATE_ZONES = tuple(range(1, 9))


def derive_test_small_split(
    registry: BuildingRegistry,
    *,
    seed: int = TEST_SMALL_SFH_SEED,
) -> dict[str, list[str]]:
    """Derive the ``test_small`` split deterministically from ``test``.

    This is the canonical definition of ``test_small``. The published
    ``splits.json`` may omit it (for example after a dataset regeneration
    that does not re-run ``make_test_small_split.py`` and re-upload the
    manifest), in which case :class:`BuildingRegistry` derives it on load so
    that ``split="test_small"`` keeps working against the published dataset.

    For each commercial building type, selects the first (sorted) test
    building in each ASHRAE climate zone 1..8. For ``SingleFamilyHouse``
    (no climate zone), selects ``TEST_SMALL_SIZE`` seeded-random test
    buildings. A climate zone with no test buildings is skipped rather than
    raising, so the registry never hard-fails on an incomplete dataset.

    .. note::
        The ``SingleFamilyHouse`` selection is seeded-random over
        ``sorted(test_ids)``, so it is reproducible only while the set of
        test IDs is unchanged. If the dataset is regenerated and SFH building
        IDs change, the same seed yields a *different* subset. This fallback
        is therefore not guaranteed reproducible across dataset
        regenerations; the canonical, reproducible source is the published
        ``splits.json``.
    """
    import random

    result: dict[str, list[str]] = {}
    for building_type in ALL_BUILDING_TYPES:
        if building_type in TYPES_WITHOUT_CLIMATE_ZONE:
            test_ids = sorted(registry.list_buildings(building_type, "test"))
            if len(test_ids) >= TEST_SMALL_SIZE:
                result[building_type] = random.Random(seed).sample(
                    test_ids, k=TEST_SMALL_SIZE
                )
            else:
                result[building_type] = test_ids
        else:
            ids: list[str] = []
            for climate_zone in _TEST_SMALL_CLIMATE_ZONES:
                zone_ids = sorted(
                    registry.list_buildings_by_climate_zone(
                        building_type, climate_zone, "test"
                    )
                )
                if zone_ids:
                    ids.append(zone_ids[0])
            result[building_type] = ids
    return result


def _validate_metadata(df: pd.DataFrame) -> None:
    """Fail fast if metadata is missing required columns."""
    if "climate_zone" not in df.columns:
        raise RuntimeError(
            "metadata.parquet is missing the 'climate_zone' column. "
            "Your HuggingFace cache is from the old dataset revision; "
            "clear ~/.cache/huggingface/hub/datasets--vtaboga--"
            "building2building_dataset/ (or re-download via "
            "huggingface_hub.snapshot_download(..., force_download=True)) "
            "and retry."
        )


@dataclass(frozen=True)
class BuildingInfo:
    """Resolved metadata for a single building."""

    building_id: str
    building_type: BuildingType
    source: str
    num_zones: int
    action_dim: int
    observation_dim: int
    net_conditioned_area_m2: float
    warmup_phases: int
    weather_file: str
    hvac_type: str
    building_dir: Path
    climate_zone: int | None


class BuildingRegistry:
    """Lazily-loaded registry backed by the unified dataset.

    Loads ``metadata.parquet`` and ``splits.json`` on first access,
    then provides fast lookups.
    """

    def __init__(self) -> None:
        self._metadata: pd.DataFrame | None = None
        self._splits: dict[str, dict[str, list[str]]] | None = None

    def _ensure_loaded(self) -> None:
        if self._metadata is None:
            meta_path = download_metadata()
            self._metadata = pd.read_parquet(meta_path)
            _validate_metadata(self._metadata)
        if self._splits is None:
            splits_path = download_splits()
            raw = json.loads(splits_path.read_text())
            if not isinstance(raw, dict):
                raise TypeError("splits.json must be a JSON object")
            self._splits = raw
            if not raw.get("test_small"):
                # Published splits.json predates / dropped the curated
                # test_small split; derive it so split="test_small" works.
                self._splits["test_small"] = derive_test_small_split(self)
                logger.info(
                    "splits.json has no 'test_small' split; derived it "
                    "deterministically from 'test'."
                )

    @property
    def metadata(self) -> pd.DataFrame:
        self._ensure_loaded()
        assert self._metadata is not None
        return self._metadata

    @property
    def splits(self) -> dict[str, dict[str, list[str]]]:
        self._ensure_loaded()
        assert self._splits is not None
        return self._splits

    def list_building_types(self) -> list[str]:
        """Return all available building types."""
        return list(ALL_BUILDING_TYPES)

    def list_buildings(
        self,
        building_type: BuildingType,
        split: SplitName = "train",
    ) -> list[str]:
        """Return building IDs for a given type and split."""
        split_data = self.splits.get(split, {})
        return split_data.get(building_type, [])

    def get_building_by_index(
        self,
        building_type: BuildingType,
        split: SplitName,
        index: int,
    ) -> BuildingInfo:
        """Select a building by split and index, return its info and path."""
        ids = self.list_buildings(building_type, split)
        if not ids:
            raise ValueError(
                f"No buildings found for type={building_type!r}, split={split!r}"
            )
        if index < 0 or index >= len(ids):
            raise IndexError(
                f"Index {index} out of range for {building_type}/{split} "
                f"(has {len(ids)} buildings)"
            )
        building_id = ids[index]
        return self.get_building_by_id(building_type, building_id)

    def get_building_by_id(
        self,
        building_type: BuildingType,
        building_id: str,
    ) -> BuildingInfo:
        """Look up a building by its ID string."""
        row = self.metadata[self.metadata["building_id"] == building_id]
        if row.empty:
            raise KeyError(f"Building {building_id!r} not found in metadata")
        r = row.iloc[0]
        building_dir = get_building_path(building_type, building_id)
        cz_raw = r.get("climate_zone")
        climate_zone: int | None = None if pd.isna(cz_raw) else int(cz_raw)
        return BuildingInfo(
            building_id=str(r["building_id"]),
            building_type=building_type,
            source=str(r.get("source", "unknown")),
            num_zones=int(r.get("num_zones", 1)),
            action_dim=int(r.get("action_dim", 0)),
            observation_dim=int(r.get("observation_dim", 0)),
            net_conditioned_area_m2=float(r.get("net_conditioned_area_m2", 0.0)),
            warmup_phases=int(r.get("warmup_phases", 1)),
            weather_file=str(r.get("weather_file", "")),
            hvac_type=str(r.get("hvac_type", "unknown")),
            building_dir=building_dir,
            climate_zone=climate_zone,
        )

    def list_buildings_by_climate_zone(
        self,
        building_type: BuildingType,
        climate_zone: int,
        split: SplitName = "train",
    ) -> list[str]:
        """Return building IDs for a given type / split filtered by ASHRAE CZ.

        Raises:
            ClimateZoneUnavailableError: If ``building_type`` is one of
                :data:`TYPES_WITHOUT_CLIMATE_ZONE` (e.g. SingleFamilyHouse),
                which has no single ASHRAE climate zone.
        """
        if building_type in TYPES_WITHOUT_CLIMATE_ZONE:
            raise ClimateZoneUnavailableError(
                f"{building_type!r} has no ASHRAE climate-zone assignment; "
                "use list_buildings(...) instead."
            )
        split_ids = set(self.list_buildings(building_type, split))
        df = self.metadata
        mask = (
            (df["building_type"] == building_type)
            & (df["building_id"].isin(split_ids))
            & (df["climate_zone"] == climate_zone)
        )
        return df.loc[mask, "building_id"].astype(str).tolist()

    def query_buildings(
        self,
        building_type: BuildingType | None = None,
        split: SplitName | None = None,
        **filters: object,
    ) -> pd.DataFrame:
        """Query the metadata with optional filters using DuckDB."""
        df = self.metadata
        if building_type is not None:
            df = df[df["building_type"] == building_type]
        if split is not None:
            valid_ids = set()
            split_data = self.splits.get(split, {})
            for ids in split_data.values():
                valid_ids.update(ids)
            df = df[df["building_id"].isin(valid_ids)]

        if filters:
            conn = duckdb.connect()
            rel = conn.from_df(df)
            for col, val in filters.items():
                if isinstance(val, str):
                    rel = rel.filter(f"lower({col}) = lower('{val}')")
                elif isinstance(val, (int, float)):
                    rel = rel.filter(f"{col} = {val}")
            df = rel.df()
            conn.close()
        return df


_registry: BuildingRegistry | None = None


def get_registry() -> BuildingRegistry:
    """Return the singleton :class:`BuildingRegistry`."""
    global _registry
    if _registry is None:
        _registry = BuildingRegistry()
    return _registry
