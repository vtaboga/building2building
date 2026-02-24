from __future__ import annotations

from typing import Literal, Sequence

from pandas import DataFrame

from b2b.sources.single_zone_houses import (
    SingleZoneHouseRowIdSplits as HydroQuebecRowIdSplits,
)
from b2b.sources.single_zone_houses import (
    building_id_from_split_index,
    filenames_for_building_id,
    select_buildings_by_split,
)


def hydroquebec_building_id_from_split_index(
    split: Literal["train", "test"],
    split_index: int,
    *,
    row_id_splits: HydroQuebecRowIdSplits | None = None,
) -> int:
    return building_id_from_split_index(
        split, split_index, row_id_splits=row_id_splits
    )


def hydroquebec_filenames_for_building_id(building_id: int) -> tuple[str, str]:
    return filenames_for_building_id(building_id)


def select_hydroquebec_buildings_by_row_id_split(
    split: Literal["train", "test"],
    *,
    row_id_splits: HydroQuebecRowIdSplits | None = None,
) -> DataFrame:
    """
    Select buildings from the HydroQuebec (big) dataset based on stored dataset row indices.

    The row ids are loaded from:
    - `b2b/sources/data/action_space_2_zone_1_train_data`
    - `b2b/sources/data/action_space_2_zone_1_test_data`
    """
    return select_buildings_by_split(split, row_id_splits=row_id_splits)
