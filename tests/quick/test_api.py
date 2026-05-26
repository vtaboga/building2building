"""Tests for building2building.api — public API functions."""
# This file pins the public API contract.
# Changes here = breaking API changes; requires a CHANGELOG.md entry.
# Marker applied automatically by conftest.py (api_contract glob).


from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from building2building.api import list_building_types


@pytest.mark.quick
class TestListBuildingTypes:
    def test_returns_all_types(self) -> None:
        types = list_building_types()
        assert isinstance(types, list)
        assert len(types) == 6
        assert "OfficeSmall" in types
        assert "SingleFamilyHouse" in types


@pytest.mark.quick
class TestListBuildings:
    def test_delegates_to_registry(self) -> None:
        mock_registry = MagicMock()
        mock_registry.list_buildings.return_value = ["OS-0001", "OS-0002"]
        with patch(
            "building2building.data.registry.get_registry",
            return_value=mock_registry,
        ):
            from building2building.api import list_buildings

            result = list_buildings("OfficeSmall", "train")
            assert result == ["OS-0001", "OS-0002"]
            mock_registry.list_buildings.assert_called_once_with("OfficeSmall", "train")

    def test_delegates_test_small_to_registry(self) -> None:
        mock_registry = MagicMock()
        mock_registry.list_buildings.return_value = ["OfficeSmall-0002"]
        with patch(
            "building2building.data.registry.get_registry",
            return_value=mock_registry,
        ):
            from building2building.api import list_buildings

            result = list_buildings("OfficeSmall", "test_small")
            assert result == ["OfficeSmall-0002"]
            mock_registry.list_buildings.assert_called_once_with(
                "OfficeSmall", "test_small"
            )
