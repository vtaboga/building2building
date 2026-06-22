from __future__ import annotations

import pandas as pd
import pytest

from building2building.data.climate_zones import TYPES_WITHOUT_CLIMATE_ZONE
from building2building.data.registry import get_registry
from building2building.data.reward_normalizers import (
    clear_reward_normalizers_cache,
    load_reward_normalizers,
    resolve_reward_normalizer,
)


@pytest.mark.release
def test_reward_normalizers_cover_metadata_buckets() -> None:
    clear_reward_normalizers_cache()
    table = load_reward_normalizers(run_period="full_year")
    registry = get_registry()
    metadata = registry.metadata[["building_id", "building_type", "climate_zone"]]

    for row in metadata.itertuples(index=False):
        building_id = str(row.building_id)
        building_type = str(row.building_type)
        by_cz = table.constants.get(building_type)
        assert by_cz is not None, f"Missing reward_normalizers entry for {building_type}"

        if building_type in TYPES_WITHOUT_CLIMATE_ZONE:
            cz_key = "cz0"
        else:
            assert not pd.isna(row.climate_zone), (
                f"Metadata climate_zone is null for {building_type}/{building_id}"
            )
            cz_key = f"cz{int(row.climate_zone)}"

        assert cz_key in by_cz, f"Missing reward normalizer bucket for {building_type}/{cz_key}"
        resolve_reward_normalizer(building_type, building_id, run_period="full_year")
