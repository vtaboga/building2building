"""Tests for climate-zone querying via the library API.

These tests exercise :mod:`building2building.data.climate_zones`,
:meth:`BuildingRegistry.list_buildings_by_climate_zone`,
:func:`building2building.api.get_climate_zone`, and the registry's hard-fail
behaviour when the ``climate_zone`` column is missing from the metadata.

The tests use the fake-dataset fixture defined in ``tests/conftest.py``; they
do not hit HuggingFace.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import building2building as b2b
from building2building.data.climate_zones import (
    TYPES_WITHOUT_CLIMATE_ZONE,
    ClimateZoneUnavailableError,
)
from building2building.data.registry import BuildingRegistry


@pytest.fixture()
def registry(
    fake_metadata: pd.DataFrame,
    fake_splits: dict[str, dict[str, list[str]]],
) -> BuildingRegistry:
    reg = BuildingRegistry()
    reg._metadata = fake_metadata
    reg._splits = fake_splits
    return reg


@pytest.mark.quick
class TestClimateZoneModule:
    def test_types_without_climate_zone_contains_sfh(self) -> None:
        assert "SingleFamilyHouse" in TYPES_WITHOUT_CLIMATE_ZONE

    def test_error_is_value_error_subclass(self) -> None:
        assert issubclass(ClimateZoneUnavailableError, ValueError)


@pytest.mark.quick
class TestListBuildingsByClimateZone:
    def test_returns_matching_ids(self, registry: BuildingRegistry) -> None:
        ids = registry.list_buildings_by_climate_zone(
            "OfficeSmall", 5, "train"
        )
        assert ids == ["OfficeSmall-0001"]

    def test_returns_empty_when_no_match(
        self, registry: BuildingRegistry
    ) -> None:
        ids = registry.list_buildings_by_climate_zone(
            "OfficeSmall", 7, "train"
        )
        assert ids == []

    def test_respects_split(self, registry: BuildingRegistry) -> None:
        train_ids = registry.list_buildings_by_climate_zone(
            "OfficeSmall", 5, "train"
        )
        test_ids = registry.list_buildings_by_climate_zone(
            "OfficeSmall", 3, "test"
        )
        assert train_ids == ["OfficeSmall-0001"]
        assert test_ids == ["OfficeSmall-0002"]

    def test_raises_for_sfh(self, registry: BuildingRegistry) -> None:
        with pytest.raises(ClimateZoneUnavailableError):
            registry.list_buildings_by_climate_zone(
                "SingleFamilyHouse", 3, "train"
            )


@pytest.mark.quick
class TestBuildingInfoClimateZone:
    def test_multizone_has_climate_zone(
        self, registry: BuildingRegistry, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "building2building.data.registry.get_building_path",
            lambda bt, bid: Path("/tmp/fake") / bt / bid,
        )
        info = registry.get_building_by_id("OfficeSmall", "OfficeSmall-0001")
        assert info.climate_zone == 5

    def test_sfh_has_null_climate_zone(
        self, registry: BuildingRegistry, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "building2building.data.registry.get_building_path",
            lambda bt, bid: Path("/tmp/fake") / bt / bid,
        )
        info = registry.get_building_by_id(
            "SingleFamilyHouse", "SingleFamilyHouse-0001"
        )
        assert info.climate_zone is None


@pytest.mark.quick
class TestMetadataColumnGuard:
    def test_missing_column_raises(
        self, fake_metadata: pd.DataFrame
    ) -> None:
        reg = BuildingRegistry()
        # Simulate an old-revision parquet without the climate_zone column by
        # dropping it and forcing _ensure_loaded to re-read via our mock path.
        stripped = fake_metadata.drop(columns=["climate_zone"])
        reg._metadata = None

        def _fake_download() -> Path:
            # unused; _ensure_loaded branches on self._metadata being None.
            raise AssertionError("should not be called")

        # Simulate by pre-populating metadata to the stripped frame *after*
        # clearing it, then directly exercising the guard.
        reg._metadata = stripped
        # Re-force the check by clearing and monkey-patching download_metadata.
        reg._metadata = None
        import building2building.data.registry as registry_mod

        orig_read_parquet = registry_mod.pd.read_parquet

        def fake_read_parquet(_path):  # noqa: ARG001
            return stripped

        registry_mod.pd.read_parquet = fake_read_parquet  # type: ignore[assignment]
        registry_mod.download_metadata = lambda: Path("/dev/null")  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="climate_zone"):
                reg._ensure_loaded()
        finally:
            registry_mod.pd.read_parquet = orig_read_parquet  # type: ignore[assignment]


@pytest.mark.quick
class TestPublicApi:
    def test_list_buildings_by_climate_zone_reexported(self) -> None:
        assert hasattr(b2b, "list_buildings_by_climate_zone")
        assert hasattr(b2b, "get_climate_zone")
        assert hasattr(b2b, "ClimateZoneUnavailableError")
        assert hasattr(b2b, "TYPES_WITHOUT_CLIMATE_ZONE")

    def test_get_climate_zone_raises_for_sfh(self) -> None:
        # No registry access needed: the check happens before the lookup.
        with pytest.raises(ClimateZoneUnavailableError):
            b2b.get_climate_zone("SingleFamilyHouse", "whatever")


# ---------------------------------------------------------------------------
# Long tests — require the real dataset on HuggingFace.
# ---------------------------------------------------------------------------


@pytest.mark.long
class TestRealDatasetClimateZones:
    """End-to-end check against the published ``vtaboga/building2building_dataset``.

    Requires network access and that Phase A's re-published parquet (with the
    ``climate_zone`` column and ``San.Diego`` rename) is available.
    """

    def test_all_multizone_rows_have_climate_zone(self) -> None:
        from building2building.data.registry import get_registry

        df = get_registry().metadata
        mz = df[~df["building_type"].isin(list(TYPES_WITHOUT_CLIMATE_ZONE))]
        assert not mz["climate_zone"].isna().any(), (
            "Every multizones row must have a climate_zone after migration"
        )
        assert mz["climate_zone"].astype(int).between(1, 8).all()

    def test_all_sfh_rows_have_null_climate_zone(self) -> None:
        from building2building.data.registry import get_registry

        df = get_registry().metadata
        sfh = df[df["building_type"] == "SingleFamilyHouse"]
        assert sfh["climate_zone"].isna().all()

    def test_no_san_deigo_typo(self) -> None:
        from building2building.data.registry import get_registry

        df = get_registry().metadata
        assert not df["weather_file"].str.contains("San.Deigo", regex=False).any()
