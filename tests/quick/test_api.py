"""Tests for building2building.api — public API functions."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from building2building.api import _infer_dataset_selection, list_building_types
from building2building.config.models import DatasetSelectionConfig


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
            mock_registry.list_buildings.assert_called_once_with(
                "OfficeSmall", "train"
            )


@pytest.mark.quick
class TestInferDatasetSelection:
    def test_default_single_zone(self) -> None:
        cfg = _infer_dataset_selection({})
        assert cfg.dataset == "single_zone_houses"
        assert cfg.mode == "split_index"
        assert cfg.split_index == 0

    def test_explicit_bldg_section(self) -> None:
        cfg = _infer_dataset_selection({
            "bldg": {
                "dataset": "multizones_reference_buildings",
                "building_type": "OfficeSmall",
                "split": "test",
                "index": 3,
            }
        })
        assert cfg.dataset == "multizones_reference_buildings"
        assert cfg.building_type == "OfficeSmall"
        assert cfg.split == "test"
        assert cfg.split_index == 3

    def test_query_mode(self) -> None:
        cfg = _infer_dataset_selection({
            "bldg": {
                "dataset": "single_zone_houses",
                "query": {"hvac_type": "baseboard"},
            }
        })
        assert cfg.mode == "metadata_query"
        assert cfg.metadata_query == {"hvac_type": "baseboard"}

    def test_non_dict_bldg_fallback(self) -> None:
        cfg = _infer_dataset_selection({"bldg": "not_a_dict"})
        assert cfg.dataset == "single_zone_houses"
        assert cfg.mode == "split_index"
