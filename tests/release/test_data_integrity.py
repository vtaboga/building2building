from __future__ import annotations

import pytest

from building2building.data.download import ALL_BUILDING_TYPES
from building2building.data.registry import get_registry


@pytest.mark.release
def test_splits_are_consistent_with_metadata() -> None:
    registry = get_registry()
    metadata_ids = set(registry.metadata["building_id"].astype(str))
    splits = registry.splits

    for split_name, by_type in splits.items():
        for building_type, ids in by_type.items():
            missing = sorted(set(ids) - metadata_ids)
            assert missing == [], (
                f"{split_name}/{building_type} contains IDs absent from metadata: "
                f"{missing[:10]}"
            )


@pytest.mark.release
def test_train_and_test_do_not_overlap() -> None:
    registry = get_registry()
    for building_type in ALL_BUILDING_TYPES:
        train_ids = set(registry.list_buildings(building_type, "train"))
        test_ids = set(registry.list_buildings(building_type, "test"))
        overlap = sorted(train_ids & test_ids)
        assert overlap == [], f"{building_type} has train/test overlap: {overlap[:10]}"


@pytest.mark.release
def test_test_small_is_subset_of_test() -> None:
    registry = get_registry()
    for building_type in ALL_BUILDING_TYPES:
        test_ids = set(registry.list_buildings(building_type, "test"))
        test_small_ids = set(registry.list_buildings(building_type, "test_small"))
        extra = sorted(test_small_ids - test_ids)
        assert extra == [], f"{building_type} has test_small IDs outside test: {extra[:10]}"
