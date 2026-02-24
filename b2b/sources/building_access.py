from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from b2b.sources import hydroquebec, multizones_reference_buildings
from b2b.sources.multizones_reference_buildings import BuildingType as MultiZonesType
from b2b.sources.single_zone_houses import (
    SplitName,
    building_id_from_split_index as single_zone_building_id_from_split_index,
    building_ids_from_split_indices as single_zone_building_ids_from_split_indices,
    sample_building_ids as sample_single_zone_building_ids,
)
from b2b.types import BuildingConfig

DatasetName = Literal["single_zone_houses", "multizones_reference_buildings"]


def sample_building_ids(
    dataset: DatasetName,
    split: SplitName,
    *,
    n: int,
    seed: int | None = None,
    replace: bool = False,
    building_type: MultiZonesType | None = None,
) -> list[int]:
    if dataset == "single_zone_houses":
        return sample_single_zone_building_ids(
            split=split,
            n=n,
            seed=seed,
            replace=replace,
        )
    if dataset == "multizones_reference_buildings":
        if building_type is None:
            raise ValueError("building_type is required for multizones sampling")
        return multizones_reference_buildings.sample_building_ids(
            building_type=building_type,
            split=split,
            n=n,
            seed=seed,
            replace=replace,
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def building_ids_from_split_indices(
    dataset: DatasetName,
    split: SplitName,
    split_indices: Sequence[int],
    *,
    building_type: MultiZonesType | None = None,
) -> list[int]:
    if dataset == "single_zone_houses":
        return single_zone_building_ids_from_split_indices(split, split_indices)
    if dataset == "multizones_reference_buildings":
        if building_type is None:
            raise ValueError("building_type is required for multizones index selection")
        return multizones_reference_buildings.building_ids_from_split_indices(
            building_type=building_type,
            split=split,
            split_indices=split_indices,
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def building_id_from_split_index(
    dataset: DatasetName,
    split: SplitName,
    split_index: int,
    *,
    building_type: MultiZonesType | None = None,
) -> int:
    if dataset == "single_zone_houses":
        return single_zone_building_id_from_split_index(split, split_index)
    if dataset == "multizones_reference_buildings":
        if building_type is None:
            raise ValueError("building_type is required for multizones index selection")
        return multizones_reference_buildings.building_id_from_split_index(
            building_type=building_type,
            split=split,
            split_index=split_index,
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def search_config(
    dataset: DatasetName,
    *,
    eplus_output_dir: Path = Path("eplus_out"),
    config: dict[str, Any] | None = None,
    building_type: MultiZonesType | None = None,
    split: SplitName | None = None,
    split_index: int | None = None,
    **query: Any,
) -> BuildingConfig:
    cfg: dict[str, Any] = dict(config) if isinstance(config, dict) else {}
    cfg.setdefault("bldg", {})
    cfg_bldg = cfg["bldg"]
    if not isinstance(cfg_bldg, dict):
        raise TypeError("config['bldg'] must be a mapping")

    if dataset == "single_zone_houses":
        cfg_bldg.setdefault("bldg", {})
        q = cfg_bldg["bldg"]
        if not isinstance(q, dict):
            raise TypeError("config['bldg']['bldg'] must be a mapping")
        q.update(query)

        if split is not None and split_index is not None:
            building_id = single_zone_building_id_from_split_index(split, split_index)
            idf_filename = f"IDFsAndSchedules/{building_id}/in.idf"
            schedule_filename = f"IDFsAndSchedules/{building_id}/in.schedules.csv"
            q.clear()
            q.update(
                {
                    "idf_filename": idf_filename,
                    "schedule_filename": schedule_filename,
                }
            )
        return hydroquebec.search_configs(
            config=cfg,
            n=1,
            eplus_output_dir=eplus_output_dir,
        )[0]

    if dataset == "multizones_reference_buildings":
        q = cfg_bldg.setdefault("bldg", {})
        if not isinstance(q, dict):
            raise TypeError("config['bldg']['bldg'] must be a mapping")
        q.update(query)
        if building_type is not None:
            q["building_type"] = building_type
        if split is not None and split_index is not None:
            if building_type is None:
                raise ValueError(
                    "building_type is required when using split/split_index for multizones"
                )
            q["building_id"] = multizones_reference_buildings.building_id_from_split_index(
                building_type=building_type,
                split=split,
                split_index=split_index,
            )
        return multizones_reference_buildings.search_config(
            config=cfg,
            eplus_output_dir=eplus_output_dir,
        )

    raise ValueError(f"Unsupported dataset: {dataset!r}")
