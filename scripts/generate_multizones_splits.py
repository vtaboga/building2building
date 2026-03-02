#!/usr/bin/env python3
"""Generate stratified train/test split files from dataset metadata.

Reads the ``metadata.csv`` produced by ``generate_dataset.py`` and creates
per-building-type JSON split files with climate-balanced train/test IDs.

Output files (per building type)::

    {output_dir}/{BuildingType}_train_data.json
    {output_dir}/{BuildingType}_test_data.json
    {output_dir}/{BuildingType}_test_small_data.json

The ``test_small`` split contains exactly one building per climate zone,
selected from the test split.

Usage::

    PYTHONPATH=. python scripts/generate_multizones_splits.py \
        --metadata dataset_v2/metadata.csv \
        --output-dir b2b/sources/data/ \
        --train-frac 0.9 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


PLACE_TO_CLIMATE_ZONE: dict[str, int] = {
    "Miami": 1,
    "Houston": 2,
    "Tampa": 2,
    "Tucson": 2,
    "Atlanta": 3,
    "ElPaso": 3,
    "SanDiego": 3,
    "SanFrancisco": 3,
    "Albuquerque": 4,
    "Baltimore": 4,
    "NewYork": 4,
    "PortAngeles": 4,
    "Seattle": 4,
    "Buffalo": 5,
    "Chicago": 5,
    "Denver": 5,
    "Vancouver": 5,
    "GreatFalls": 6,
    "Rochester": 6,
    "Duluth": 7,
    "InternationalFalls": 7,
    "Fairbanks": 8,
}


def stratified_split(
    ids: list[int],
    groups: list[str],
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Split *ids* into train/test, stratified by *groups*.

    Within each unique group value, ``train_frac`` of the IDs go to train and
    the rest to test.  Guarantees every group appears in both sets (as long as
    the group has >= 2 members).
    """
    ids_arr = np.array(ids)
    groups_arr = np.array(groups)
    train_ids: list[int] = []
    test_ids: list[int] = []

    for group in sorted(set(groups)):
        mask = groups_arr == group
        group_ids = ids_arr[mask].copy()
        rng.shuffle(group_ids)
        n_train = max(1, int(round(len(group_ids) * train_frac)))
        train_ids.extend(int(x) for x in group_ids[:n_train])
        test_ids.extend(int(x) for x in group_ids[n_train:])

    return sorted(train_ids), sorted(test_ids)


def pick_one_per_climate_zone(
    test_ids: list[int],
    places: pd.Series,
    rng: np.random.Generator,
) -> list[int]:
    """Pick one test building per climate zone.

    *places* is a Series indexed by building_id with place names as values.
    """
    test_places = places.loc[test_ids]
    cz_series = test_places.map(PLACE_TO_CLIMATE_ZONE)

    small_ids: list[int] = []
    for cz in sorted(cz_series.unique()):
        candidates = cz_series[cz_series == cz].index.tolist()
        chosen = int(rng.choice(candidates))
        small_ids.append(chosen)

    return sorted(small_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.csv from generate_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("b2b/sources/data"),
        help="Directory to write split JSON files (default: b2b/sources/data)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.9,
        help="Fraction of buildings in the training set (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    metadata_path: Path = args.metadata
    output_dir: Path = args.output_dir
    train_frac: float = args.train_frac
    seed: int = args.seed

    if not metadata_path.exists():
        parser.error(f"Metadata file not found: {metadata_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    log.info("Loaded %d rows from %s", len(df), metadata_path)

    rng = np.random.default_rng(seed)

    for btype, group_df in df.groupby("building_type"):
        ids = group_df["building_id"].tolist()
        places = group_df["place"].tolist()
        train_ids, test_ids = stratified_split(ids, places, train_frac, rng)

        place_by_id = group_df.set_index("building_id")["place"]
        test_small_ids = pick_one_per_climate_zone(test_ids, place_by_id, rng)

        for split_name, split_ids in [
            ("train", train_ids),
            ("test", test_ids),
            ("test_small", test_small_ids),
        ]:
            out_path = output_dir / f"{btype}_{split_name}_data.json"
            out_path.write_text(
                json.dumps(split_ids, indent=2) + "\n", encoding="utf-8"
            )

        n_places_train = len(set(group_df.loc[group_df["building_id"].isin(train_ids), "place"]))
        n_places_test = len(set(group_df.loc[group_df["building_id"].isin(test_ids), "place"]))
        test_small_czs = sorted(
            PLACE_TO_CLIMATE_ZONE[place_by_id[bid]]
            for bid in test_small_ids
        )
        log.info(
            "  %s: %d train (%d places), %d test (%d places), "
            "%d test_small (CZs: %s)",
            btype,
            len(train_ids),
            n_places_train,
            len(test_ids),
            n_places_test,
            len(test_small_ids),
            test_small_czs,
        )

    log.info("Split files written to %s", output_dir)


if __name__ == "__main__":
    main()
