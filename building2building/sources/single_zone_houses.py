from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pandas import DataFrame

from building2building.sources import residential

SplitName = Literal["train", "test"]


@dataclass(frozen=True)
class SingleZoneHouseRowIdSplits:
    train_row_ids: list[int]
    test_row_ids: list[int]

    @classmethod
    def load_default(cls) -> "SingleZoneHouseRowIdSplits":
        data_dir = Path(__file__).resolve().parent / "data"
        train_path = data_dir / "action_space_2_zone_1_train_data.json"
        test_path = data_dir / "action_space_2_zone_1_test_data.json"
        return cls(
            train_row_ids=_load_json_int_list(train_path),
            test_row_ids=_load_json_int_list(test_path),
        )

    # Backward-compatible name used in several modules/tests.
    @classmethod
    def load_from_action_space_2_zone_1(cls) -> "SingleZoneHouseRowIdSplits":
        return cls.load_default()

    def save_json(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "action_space_2_zone_1_train_row_ids.json").write_text(
            json.dumps(self.train_row_ids), encoding="utf-8"
        )
        (out_dir / "action_space_2_zone_1_test_row_ids.json").write_text(
            json.dumps(self.test_row_ids), encoding="utf-8"
        )


def _load_json_int_list(path: Path) -> list[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"Expected a list in {path}, got {type(obj).__name__}")
    if not all(isinstance(x, int) for x in obj):
        bad_types = sorted({type(x).__name__ for x in obj if not isinstance(x, int)})
        raise TypeError(f"Expected list[int] in {path}, got non-ints: {bad_types}")
    return obj


def split_row_ids(
    split: SplitName,
    *,
    row_id_splits: SingleZoneHouseRowIdSplits | None = None,
) -> list[int]:
    splits = row_id_splits or SingleZoneHouseRowIdSplits.load_default()
    return list(splits.train_row_ids if split == "train" else splits.test_row_ids)


def building_id_from_split_index(
    split: SplitName,
    split_index: int,
    *,
    row_id_splits: SingleZoneHouseRowIdSplits | None = None,
) -> int:
    if not isinstance(split_index, int):
        raise TypeError(f"split_index must be int, got {type(split_index).__name__}")
    if split_index < 0:
        raise ValueError(f"split_index must be >= 0, got {split_index}")
    ids = split_row_ids(split, row_id_splits=row_id_splits)
    if split_index >= len(ids):
        raise IndexError(
            f"split_index={split_index} out of range for split={split!r} (len={len(ids)})"
        )
    dataset_row_index = ids[split_index]
    if dataset_row_index < 0:
        raise ValueError(
            f"Expected residential dataset row index >= 0, got {dataset_row_index}"
        )
    # Hydro-Quebec dataset row indices are 0-based, filenames are 1-based.
    return int(dataset_row_index) + 1


def building_ids_from_split_indices(
    split: SplitName,
    split_indices: Sequence[int],
    *,
    row_id_splits: SingleZoneHouseRowIdSplits | None = None,
) -> list[int]:
    out: list[int] = []
    for idx in split_indices:
        out.append(
            building_id_from_split_index(split, int(idx), row_id_splits=row_id_splits)
        )
    return out


def sample_building_ids(
    split: SplitName,
    n: int,
    *,
    seed: int | None = None,
    replace: bool = False,
    row_id_splits: SingleZoneHouseRowIdSplits | None = None,
) -> list[int]:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be int >= 0, got {n!r}")
    ids = split_row_ids(split, row_id_splits=row_id_splits)
    building_ids = [int(row_id) + 1 for row_id in ids]
    rng = random.Random(seed)
    if n == 0:
        return []
    if replace:
        return [rng.choice(building_ids) for _ in range(n)]
    if n > len(building_ids):
        raise ValueError(
            f"Cannot sample n={n} without replacement from only {len(building_ids)} ids"
        )
    return rng.sample(building_ids, k=n)


def filenames_for_building_id(building_id: int) -> tuple[str, str]:
    if not isinstance(building_id, int):
        raise TypeError(f"building_id must be int, got {type(building_id).__name__}")
    if building_id < 1:
        raise ValueError(f"building_id must be >= 1, got {building_id}")
    return (
        f"IDFsAndSchedules/{building_id}/in.idf",
        f"IDFsAndSchedules/{building_id}/in.schedules.csv",
    )


def select_buildings_by_split(
    split: SplitName,
    *,
    row_id_splits: SingleZoneHouseRowIdSplits | None = None,
) -> DataFrame:
    row_ids = split_row_ids(split, row_id_splits=row_id_splits)
    df = residential.search_buildings()
    if not row_ids:
        return df.iloc[0:0]
    ilocs: list[int] = []
    for rid in row_ids:
        if rid < 0:
            raise ValueError(f"Expected dataset row index >= 0, got {rid}")
        ilocs.append(int(rid))
    return df.iloc[ilocs]
