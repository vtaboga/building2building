from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from b2b.config.models import DatasetName, DatasetSelectionConfig, SplitName
from b2b.sources import hydroquebec, multizones_reference_buildings
from b2b.sources.multizones_reference_buildings import BuildingType as MultiZonesType
from b2b.sources.single_zone_houses import (
    building_id_from_split_index as single_zone_building_id_from_split_index,
)
from b2b.sources.single_zone_houses import (
    building_ids_from_split_indices as single_zone_building_ids_from_split_indices,
)
from b2b.sources.single_zone_houses import sample_building_ids as sample_single_zone_building_ids
from b2b.sources.single_zone_houses import split_row_ids as single_zone_split_row_ids
from b2b.types import BuildingConfig


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


def query_building_ids(
    selection: DatasetSelectionConfig,
) -> list[int]:
    if selection.mode != "metadata_query":
        raise ValueError("query_building_ids expects metadata_query mode")
    if selection.dataset == "single_zone_houses":
        rows = hydroquebec.search_buildings(**selection.metadata_query)
        split_rows = (
            set(single_zone_split_row_ids(selection.split))
            if selection.split is not None
            else None
        )
        out: list[int] = []
        for _, row in rows.iterrows():
            row_idx = int(row.name)
            if split_rows is not None and row_idx not in split_rows:
                continue
            building_id = row_idx + 1
            if building_id not in out:
                out.append(building_id)
        return out[: selection.sample_size]

    if selection.dataset == "multizones_reference_buildings":
        if selection.building_type is None:
            raise ValueError(
                "dataset_selection.building_type is required for multizones queries"
            )
        split_ids = (
            set(
                multizones_reference_buildings.load_split_ids(
                    selection.building_type, selection.split
                )
            )
            if selection.split is not None
            else None
        )
        rows = multizones_reference_buildings.search_buildings(
            building_type=selection.building_type,
            **selection.metadata_query,
        )
        out: list[int] = []
        for _, row in rows.iterrows():
            building_id = int(row.building_id)
            if split_ids is not None and building_id not in split_ids:
                continue
            if building_id not in out:
                out.append(building_id)
        return out[: selection.sample_size]

    raise ValueError(f"Unsupported dataset: {selection.dataset!r}")


def select_building_ids(selection: DatasetSelectionConfig) -> list[int]:
    if selection.mode == "building_id":
        if selection.building_id is None:
            raise ValueError(
                "dataset_selection.building_id must be provided in building_id mode"
            )
        return [int(selection.building_id)]
    if selection.mode == "split_index":
        if selection.split is None:
            raise ValueError("dataset_selection.split is required in split_index mode")
        return [
            building_id_from_split_index(
                dataset=selection.dataset,
                split=selection.split,
                split_index=selection.split_index,
                building_type=selection.building_type,
            )
        ]
    if selection.mode == "split_indices":
        if selection.split is None:
            raise ValueError("dataset_selection.split is required in split_indices mode")
        return building_ids_from_split_indices(
            dataset=selection.dataset,
            split=selection.split,
            split_indices=selection.split_indices,
            building_type=selection.building_type,
        )
    if selection.mode == "random":
        if selection.split is None:
            raise ValueError("dataset_selection.split is required in random mode")
        return sample_building_ids(
            dataset=selection.dataset,
            split=selection.split,
            n=selection.sample_size,
            seed=selection.seed,
            replace=selection.replace,
            building_type=selection.building_type,
        )
    if selection.mode == "metadata_query":
        return query_building_ids(selection)
    raise ValueError(f"Unsupported selection mode: {selection.mode!r}")


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
            q.clear()
            q.update(
                {
                    "idf_filename": f"IDFsAndSchedules/{building_id}/in.idf",
                    "schedule_filename": f"IDFsAndSchedules/{building_id}/in.schedules.csv",
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
