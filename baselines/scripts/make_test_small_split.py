#!/usr/bin/env python3
"""Generate an updated splits.json with a deterministic test_small split.

The generated file is intended for manual review and upload to the
``vtaboga/building2building_dataset`` Hugging Face dataset as ``splits.json``.
It preserves the existing ``train`` and ``test`` entries and adds
``test_small`` with 8 IDs per building type:

* commercial types: one test building per climate zone 1..8
* SingleFamilyHouse: 8 random test buildings with a fixed seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypeAlias

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SplitManifest: TypeAlias = dict[str, dict[str, list[str]]]

BUILDING_TYPES: tuple[str, ...] = (
    "SingleFamilyHouse",
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
)
COMMERCIAL_BUILDING_TYPES: tuple[str, ...] = (
    "OfficeSmall",
    "OfficeMedium",
    "RetailStandalone",
    "RestaurantFastFood",
    "Warehouse",
)
CLIMATE_ZONES = tuple(range(1, 9))


def _b2b_module() -> Any:
    import building2building as b2b

    return b2b


def _load_current_splits() -> SplitManifest:
    from huggingface_hub import hf_hub_download

    splits_path = Path(
        hf_hub_download(
            repo_id="vtaboga/building2building_dataset",
            filename="splits.json",
            repo_type="dataset",
            revision="main",
        )
    )
    raw = json.loads(splits_path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"Expected splits.json to contain an object, got {type(raw)}")
    return raw


def build_test_small_split(seed: int) -> dict[str, list[str]]:
    """Build the deterministic test_small split mapping.

    Delegates to the canonical
    :func:`building2building.data.registry.derive_test_small_split` so the
    manual-upload path and the registry's in-code fallback never diverge.
    The downstream :func:`_validate_test_small` enforces the "exactly 8 per
    type" contract (the library derivation is lenient and skips empty
    climate zones).
    """
    from building2building.data.registry import (
        derive_test_small_split,
        get_registry,
    )

    derived = derive_test_small_split(get_registry(), seed=seed)
    return {building_type: derived[building_type] for building_type in BUILDING_TYPES}


def _validate_test_small(
    splits: SplitManifest,
    test_small: dict[str, list[str]],
) -> None:
    test_split = splits.get("test", {})
    if not isinstance(test_split, dict):
        raise TypeError("splits.json must contain a top-level 'test' object")

    for building_type in BUILDING_TYPES:
        test_ids = test_split.get(building_type, [])
        if len(test_ids) < 100:
            raise ValueError(
                f"Expected at least 100 test buildings for {building_type}, "
                f"got {len(test_ids)}"
            )

        small_ids = test_small.get(building_type, [])
        if len(small_ids) != 8:
            raise ValueError(
                f"Expected exactly 8 test_small IDs for {building_type}, "
                f"got {len(small_ids)}"
            )
        if len(set(small_ids)) != 8:
            raise ValueError(f"Duplicate test_small IDs for {building_type}: {small_ids}")

        unknown_ids = sorted(set(small_ids).difference(test_ids))
        if unknown_ids:
            raise ValueError(
                f"test_small IDs for {building_type} are not in test split: "
                f"{unknown_ids}"
            )

    b2b = _b2b_module()
    for building_type in COMMERCIAL_BUILDING_TYPES:
        for building_id, climate_zone in zip(
            test_small[building_type], CLIMATE_ZONES, strict=True
        ):
            zone_ids = set(
                b2b.list_buildings_by_climate_zone(
                    building_type, climate_zone, split="test"
                )
            )
            if building_id not in zone_ids:
                raise ValueError(
                    f"{building_id} is not in {building_type} test climate "
                    f"zone {climate_zone}"
                )


def _print_summary(test_small: dict[str, list[str]]) -> None:
    print("Selected test_small IDs:")
    for building_type in BUILDING_TYPES:
        print(f"\n{building_type}:")
        if building_type in COMMERCIAL_BUILDING_TYPES:
            for climate_zone, building_id in zip(
                CLIMATE_ZONES, test_small[building_type], strict=True
            ):
                print(f"  climate_zone={climate_zone}: {building_id}")
        else:
            for building_id in test_small[building_type]:
                print(f"  {building_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate splits.json with an added test_small split."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/splits_with_test_small.json"),
        help="Path for the generated splits manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for selecting SingleFamilyHouse test_small IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = _load_current_splits()
    test_small = build_test_small_split(seed=args.seed)
    _validate_test_small(splits, test_small)

    updated_splits = dict(splits)
    updated_splits["test_small"] = test_small

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(updated_splits, indent=2) + "\n")
    _print_summary(test_small)
    print(f"\nWrote updated splits manifest to {args.output}")


if __name__ == "__main__":
    main()
