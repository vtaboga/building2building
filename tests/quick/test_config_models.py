"""Tests for building2building.config.models — DatasetSelectionConfig, EnvBuildConfig."""

from __future__ import annotations

import pytest

from building2building.config.models import (
    BuildingType,
    DatasetSelectionConfig,
    EnvBuildConfig,
)


@pytest.mark.quick
class TestDatasetSelectionConfig:
    def test_defaults(self) -> None:
        cfg = DatasetSelectionConfig.from_dict({"dataset": "single_zone_houses"})
        assert cfg.dataset == "single_zone_houses"
        assert cfg.split == "train"
        assert cfg.mode == "split_index"
        assert cfg.split_index == 0

    def test_multizones_dataset(self) -> None:
        cfg = DatasetSelectionConfig.from_dict({
            "dataset": "multizones_reference_buildings",
            "mode": "building_id",
            "building_id": 42,
            "building_type": "OfficeSmall",
        })
        assert cfg.dataset == "multizones_reference_buildings"
        assert cfg.mode == "building_id"
        assert cfg.building_id == 42

    def test_invalid_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="dataset_selection.dataset"):
            DatasetSelectionConfig.from_dict({"dataset": "imagenet"})

    def test_frozen(self) -> None:
        cfg = DatasetSelectionConfig(dataset="single_zone_houses")
        with pytest.raises(AttributeError):
            cfg.dataset = "multizones_reference_buildings"  # type: ignore[misc]


@pytest.mark.quick
class TestEnvBuildConfig:
    def test_from_dict_minimal(self) -> None:
        raw = {
            "dataset_selection": {"dataset": "single_zone_houses"},
            "task": {},
            "reward": {"reward_type": "DeadbandRewardConfig"},
        }
        cfg = EnvBuildConfig.from_dict(raw)
        assert cfg.dataset_selection.dataset == "single_zone_houses"
        assert cfg.env_max_steps is None
        assert cfg.expose_heating_only_zones is True

    def test_from_dict_with_max_steps(self) -> None:
        raw = {
            "dataset_selection": {"dataset": "single_zone_houses"},
            "task": {"run_period": "winter"},
            "reward": {"reward_type": "BaseRewardConfig"},
            "env_max_steps": 1000,
        }
        cfg = EnvBuildConfig.from_dict(raw)
        assert cfg.env_max_steps == 1000
        assert cfg.task.run_period.name == "winter"

    def test_frozen(self) -> None:
        raw = {
            "dataset_selection": {"dataset": "single_zone_houses"},
            "task": {},
            "reward": {"reward_type": "DeadbandRewardConfig"},
        }
        cfg = EnvBuildConfig.from_dict(raw)
        with pytest.raises(AttributeError):
            cfg.env_max_steps = 500  # type: ignore[misc]


@pytest.mark.quick
class TestBuildingType:
    def test_known_types(self) -> None:
        from building2building.data.download import ALL_BUILDING_TYPES

        expected = {
            "Warehouse",
            "RetailStandalone",
            "RestaurantFastFood",
            "OfficeMedium",
            "OfficeSmall",
            "SingleFamilyHouse",
        }
        assert set(ALL_BUILDING_TYPES) == expected

    def test_no_hotel_small(self) -> None:
        from building2building.data.download import ALL_BUILDING_TYPES

        assert "HotelSmall" not in ALL_BUILDING_TYPES
