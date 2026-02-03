from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pandas import DataFrame

from building2building.sources import hydroquebec


@dataclass(frozen=True)
class HydroQuebecRowIdSplits:
    train_row_ids: list[int]
    test_row_ids: list[int]

    @classmethod
    def load_from_action_space_2_zone_1(cls) -> "HydroQuebecRowIdSplits":
        data_dir = Path(__file__).resolve().parent / "sources" / "data"
        train_path = data_dir / "action_space_2_zone_1_train_data"
        test_path = data_dir / "action_space_2_zone_1_test_data"
        return cls(
            train_row_ids=_load_pickled_int_list(train_path),
            test_row_ids=_load_pickled_int_list(test_path),
        )

    def save_json(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "action_space_2_zone_1_train_row_ids.json").write_text(
            json.dumps(self.train_row_ids), encoding="utf-8"
        )
        (out_dir / "action_space_2_zone_1_test_row_ids.json").write_text(
            json.dumps(self.test_row_ids), encoding="utf-8"
        )


def _load_pickled_int_list(path: Path) -> list[int]:
    obj = pickle.loads(path.read_bytes())
    if not isinstance(obj, list):
        raise TypeError(f"Expected a list in {path}, got {type(obj).__name__}")
    if not all(isinstance(x, int) for x in obj):
        bad_types = sorted({type(x).__name__ for x in obj if not isinstance(x, int)})
        raise TypeError(f"Expected list[int] in {path}, got non-ints: {bad_types}")
    return obj


def hydroquebec_building_id_from_split_index(
    split: Literal["train", "test"],
    split_index: int,
    *,
    row_id_splits: HydroQuebecRowIdSplits | None = None,
) -> int:
    """
    Return a HydroQuebec *building id* (1-based) from a split and index.

    The returned id corresponds to filenames like:
    - `IDFsAndSchedules/<id>/in.idf`
    - `IDFsAndSchedules/<id>/in.schedules.csv`
    """
    if not isinstance(split_index, int):
        raise TypeError(f"split_index must be int, got {type(split_index).__name__}")
    if split_index < 0:
        raise ValueError(f"split_index must be >= 0, got {split_index}")

    splits = row_id_splits or HydroQuebecRowIdSplits.load_from_action_space_2_zone_1()
    ids: Sequence[int] = splits.train_row_ids if split == "train" else splits.test_row_ids
    if split_index >= len(ids):
        raise IndexError(
            f"split_index={split_index} out of range for split={split!r} (len={len(ids)})"
        )
    building_id = ids[split_index]
    if building_id < 1:
        raise ValueError(f"Expected HydroQuebec building id >= 1, got {building_id}")
    return int(building_id)


def hydroquebec_filenames_for_building_id(building_id: int) -> tuple[str, str]:
    if not isinstance(building_id, int):
        raise TypeError(f"building_id must be int, got {type(building_id).__name__}")
    if building_id < 1:
        raise ValueError(f"building_id must be >= 1, got {building_id}")
    idf = f"IDFsAndSchedules/{building_id}/in.idf"
    schedules = f"IDFsAndSchedules/{building_id}/in.schedules.csv"
    return idf, schedules


def select_hydroquebec_buildings_by_row_id_split(
    split: Literal["train", "test"],
    *,
    row_id_splits: HydroQuebecRowIdSplits | None = None,
) -> DataFrame:
    """
    Select buildings from the HydroQuebec (big) dataset based on stored building ids.

    The row ids are loaded from:
    - `building2building/sources/data/action_space_2_zone_1_train_data`
    - `building2building/sources/data/action_space_2_zone_1_test_data`
    """
    splits = row_id_splits or HydroQuebecRowIdSplits.load_from_action_space_2_zone_1()
    row_ids: Sequence[int] = splits.train_row_ids if split == "train" else splits.test_row_ids

    # Note: `hydroquebec.search_buildings()` returns the whole dataset (a few 10k rows),
    # with a default 0..N-1 index. The stored ids are 1-based and correspond to
    # filenames `IDFsAndSchedules/<id>/...`, so they map to `.iloc[id - 1]`.
    df = hydroquebec.search_buildings()
    if not row_ids:
        return df.iloc[0:0]

    ilocs: list[int] = []
    for rid in row_ids:
        if rid < 1:
            raise ValueError(f"Expected row id >= 1, got {rid}")
        ilocs.append(int(rid) - 1)
    return df.iloc[ilocs]
