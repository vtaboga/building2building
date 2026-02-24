from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from b2b.sources import building_access
from b2b.sources.single_zone_houses import (
    SingleZoneHouseRowIdSplits,
    building_id_from_split_index,
    building_ids_from_split_indices,
    sample_building_ids,
)
from b2b.types import BaseRewardConfig, BuildingConfig


def test_single_zone_building_id_mapping() -> None:
    splits = SingleZoneHouseRowIdSplits(
        train_row_ids=[0, 2, 5],
        test_row_ids=[1, 3],
    )
    assert building_id_from_split_index("train", 0, row_id_splits=splits) == 1
    assert building_id_from_split_index("train", 2, row_id_splits=splits) == 6
    assert building_ids_from_split_indices(
        "test", [0, 1], row_id_splits=splits
    ) == [2, 4]


def test_single_zone_sampling_reproducible() -> None:
    splits = SingleZoneHouseRowIdSplits(
        train_row_ids=[0, 1, 2, 3, 4],
        test_row_ids=[10, 11],
    )
    a = sample_building_ids("train", 3, seed=123, row_id_splits=splits)
    b = sample_building_ids("train", 3, seed=123, row_id_splits=splits)
    assert a == b
    assert len(a) == 3


def test_building_access_multizones_requires_building_type() -> None:
    with pytest.raises(ValueError):
        building_access.sample_building_ids(
            "multizones_reference_buildings",
            "train",
            n=1,
        )


@patch("b2b.sources.building_access.hydroquebec.search_configs")
def test_building_access_search_config_single_zone_from_split(
    mock_search_configs: MagicMock,
) -> None:
    mock_cfg = BuildingConfig(
        path_to_building=Path("a"),
        path_to_weather=Path("b"),
        reward_config=BaseRewardConfig(0.0),
        eplus_output_dir=Path("eplus"),
        warmup_phases=1,
        area=1.0,
        hvac_equipment=[],
    )
    mock_search_configs.return_value = [mock_cfg]

    _ = building_access.search_config(
        "single_zone_houses",
        split="train",
        split_index=0,
        config={"bldg": {"bldg": {}}, "reward": {"reward_type": None}},
    )

    assert mock_search_configs.called
    kwargs = mock_search_configs.call_args.kwargs
    cfg = kwargs["config"]
    assert isinstance(cfg, dict)
    bldg_section = cfg["bldg"]["bldg"]
    assert "idf_filename" in bldg_section
    assert "schedule_filename" in bldg_section


@patch("b2b.sources.building_access.multizones_reference_buildings.search_config")
def test_building_access_search_config_multizones_split(
    mock_search_config: MagicMock,
) -> None:
    mock_cfg = BuildingConfig(
        path_to_building=Path("a"),
        path_to_weather=Path("b"),
        reward_config=BaseRewardConfig(0.0),
        eplus_output_dir=Path("eplus"),
        warmup_phases=1,
        area=1.0,
        hvac_equipment=[],
    )
    mock_search_config.return_value = mock_cfg

    _ = building_access.search_config(
        "multizones_reference_buildings",
        building_type="OfficeSmall",
        split="train",
        split_index=0,
        config={"bldg": {"bldg": {}}, "reward": {"reward_type": None}},
    )

    assert mock_search_config.called
