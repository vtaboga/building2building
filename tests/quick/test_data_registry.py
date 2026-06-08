"""Tests for building2building.data.registry — BuildingRegistry with fake data."""
# This file pins the public API contract.
# Changes here = breaking API changes; requires a CHANGELOG.md entry.
# Marker applied automatically by conftest.py (api_contract glob).


from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from building2building.data.registry import BuildingInfo, BuildingRegistry


@pytest.fixture()
def registry(
    fake_dataset_dir: Path,
    fake_metadata: pd.DataFrame,
    fake_splits: dict[str, dict[str, list[str]]],
) -> BuildingRegistry:
    """Create a BuildingRegistry pre-loaded with fixture data."""
    reg = BuildingRegistry()
    reg._metadata = fake_metadata
    reg._splits = fake_splits
    return reg


@pytest.mark.quick
class TestBuildingRegistryListings:
    def test_list_building_types(self, registry: BuildingRegistry) -> None:
        types = registry.list_building_types()
        assert "OfficeSmall" in types
        assert "SingleFamilyHouse" in types
        assert len(types) == 6

    def test_list_buildings_train(self, registry: BuildingRegistry) -> None:
        ids = registry.list_buildings("OfficeSmall", "train")
        assert ids == ["OfficeSmall-0001"]

    def test_list_buildings_test(self, registry: BuildingRegistry) -> None:
        ids = registry.list_buildings("OfficeSmall", "test")
        assert ids == ["OfficeSmall-0002"]

    def test_list_buildings_test_small(self, registry: BuildingRegistry) -> None:
        ids = registry.list_buildings("OfficeSmall", "test_small")
        assert ids == ["OfficeSmall-0002"]

    def test_list_buildings_empty_split(self, registry: BuildingRegistry) -> None:
        ids = registry.list_buildings("RetailStandalone", "train")
        assert ids == []


@pytest.mark.quick
class TestBuildingRegistryLookups:
    def test_get_building_by_index(
        self, registry: BuildingRegistry, fake_dataset_dir: Path
    ) -> None:
        with patch(
            "building2building.data.download.download_building_type",
            return_value=fake_dataset_dir / "OfficeSmall",
        ):
            info = registry.get_building_by_index("OfficeSmall", "train", 0)
            assert isinstance(info, BuildingInfo)
            assert info.building_id == "OfficeSmall-0001"
            assert info.building_type == "OfficeSmall"
            assert info.num_zones == 5

    def test_get_building_by_index_out_of_range(
        self, registry: BuildingRegistry
    ) -> None:
        with pytest.raises(IndexError, match="out of range"):
            registry.get_building_by_index("OfficeSmall", "train", 99)

    def test_get_building_by_index_empty_split(
        self, registry: BuildingRegistry
    ) -> None:
        with pytest.raises(ValueError, match="No buildings found"):
            registry.get_building_by_index("RetailStandalone", "train", 0)

    def test_get_building_by_id(
        self, registry: BuildingRegistry, fake_dataset_dir: Path
    ) -> None:
        with patch(
            "building2building.data.download.download_building_type",
            return_value=fake_dataset_dir / "Warehouse",
        ):
            info = registry.get_building_by_id("Warehouse", "Warehouse-0001")
            assert info.building_id == "Warehouse-0001"
            assert info.net_conditioned_area_m2 == 4800.0

    def test_get_building_by_id_not_found(self, registry: BuildingRegistry) -> None:
        with pytest.raises(KeyError, match="not found in metadata"):
            registry.get_building_by_id("OfficeSmall", "OfficeSmall-9999")


@pytest.mark.quick
class TestBuildingRegistryQuery:
    def test_query_by_building_type(self, registry: BuildingRegistry) -> None:
        df = registry.query_buildings(building_type="OfficeSmall")
        assert len(df) == 2
        assert set(df["building_id"]) == {"OfficeSmall-0001", "OfficeSmall-0002"}

    def test_query_by_split(self, registry: BuildingRegistry) -> None:
        df = registry.query_buildings(split="train")
        expected_ids = {"OfficeSmall-0001", "SingleFamilyHouse-0001", "Warehouse-0001"}
        assert set(df["building_id"]) == expected_ids

    def test_query_by_test_small_split(self, registry: BuildingRegistry) -> None:
        df = registry.query_buildings(split="test_small")
        assert set(df["building_id"]) == {"OfficeSmall-0002"}

    def test_query_with_filter(self, registry: BuildingRegistry) -> None:
        df = registry.query_buildings(hvac_type="baseboard")
        assert len(df) == 1
        assert df.iloc[0]["building_id"] == "SingleFamilyHouse-0001"
