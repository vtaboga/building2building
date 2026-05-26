"""Regenerate a slice of the HuggingFace dataset under the current pipeline.

Phase M, item M2 (see TODO.md and notes.md § "OfficeMedium OA-mixer
fix").

The HuggingFace dataset ``vtaboga/building2building_dataset`` stores
post-``make_controllable`` artefacts: per-building ``building.epjson``,
``equipment.json``, ``metadata.json`` and a unified ``metadata.parquet``
indexing all buildings.  Whenever the actuator emission in
``building2building.pipeline.actuators`` changes, every downstream
artefact for the affected building types must be re-derived through
the new pipeline.  This module does that for any subset of building
types, one building per process invocation, designed for trivial
parallelization via Slurm arrays.

Canonical reproduction (single machine, all buildings of one type):

    python -m building2building.pipeline.regen_dataset \\
        --building-type OfficeMedium \\
        --output-dir <staging>

For Slurm array parallelization, see
``building2building/pipeline/scripts/regen_officemedium.sh``.

Output layout under ``<staging>``::

    <staging>/
        OfficeMedium/                 (or other building-type)
            OfficeMedium-4001/
                building.epjson
                equipment.json
                metadata.json
            ...
        metadata.parquet              (rewritten with new action_dim)
        splits.json                   (copied unchanged)

The staging directory is what you then ``huggingface-cli upload`` to
``vtaboga/building2building_dataset@main``.  This module never pushes
to HF.

Per AGENTS.md: net_conditioned_area and warmup_phases do not depend on
the actuator inventory; both are read from the existing per-building
``metadata.json`` rather than re-computed, which skips the per-building
EnergyPlus discovery run and reduces the regen time by ~10x.  If the
upstream pipeline ever changes RunPeriod, sizing, or building geometry,
that assumption breaks and ``--rerun-discovery`` must be passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from cattrs import unstructure

from building2building.data.download import (
    ALL_BUILDING_TYPES,
    BuildingType,
    download_building_type,
    download_metadata,
    download_splits,
)
from building2building.env import STORE_PATH
from building2building.pipeline.discovery import extract_discovery_metadata
from building2building.sources.multizones_reference_buildings import (
    _build_control_derivation,
    dataset_zip,
)
from building2building.store import Constant, ExtractFromZip, realize

logger = logging.getLogger(__name__)


# Building-types that this regen script supports.  ``SingleFamilyHouse``
# has a different source pipeline (see building2building/sources/
# residential.py) and is intentionally out of scope here.
SUPPORTED_BUILDING_TYPES: tuple[BuildingType, ...] = (
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
)


def _resolve_source_building_id(existing_dir: Path) -> int:
    """Read the source-IDF building_id from the existing metadata.json."""
    meta_path = existing_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Cannot resolve source_building_id: {meta_path} does not exist"
        )
    meta = json.loads(meta_path.read_text())
    if "source_building_id" not in meta:
        raise KeyError(
            f"{meta_path} has no 'source_building_id' field; cannot map "
            f"back to the raw multizones_reference_buildings.zip entry."
        )
    return int(meta["source_building_id"])


def regen_one_building(
    building_type: BuildingType,
    building_id: str,
    existing_dir: Path,
    out_root: Path,
    *,
    rerun_discovery: bool = False,
) -> dict:
    """Regenerate one building's controllable artefacts.

    Reads the source epJSON from ``multizones_reference_buildings.zip``,
    runs the current pipeline, and writes ``building.epjson``,
    ``equipment.json``, and ``metadata.json`` under
    ``out_root / building_type / building_id /``.

    Returns a small summary dict suitable for aggregation into
    ``metadata.parquet`` (``num_actuators``, etc.).
    """
    target_dir = out_root / building_type / building_id
    target_dir.mkdir(parents=True, exist_ok=True)

    source_building_id = _resolve_source_building_id(existing_dir)
    epjson_filename = f"{source_building_id}.epJSON"

    root_zip = dataset_zip()
    derivation = _build_control_derivation(
        root_zip=root_zip,
        epjson_filename=epjson_filename,
        run_period_name="full_year",
    )
    epjson_path, equipment_list = realize(STORE_PATH.get(), derivation)

    # Copy the realized artefacts into the per-building output dir.
    shutil.copy(epjson_path, target_dir / "building.epjson")
    with open(target_dir / "equipment.json", "w") as f:
        json.dump(unstructure(list(equipment_list)), f, indent=4)

    # net_conditioned_area and warmup_phases do not depend on the
    # actuator inventory (per notes.md § Q4); either copy from the
    # existing metadata.json or recompute via the discovery sim.
    if rerun_discovery:
        # Locate the weather file shipped alongside the existing
        # building dir; its name is stable across regens.
        epws = sorted(existing_dir.glob("*.epw"))
        if not epws:
            raise FileNotFoundError(
                f"--rerun-discovery requested but no .epw found in "
                f"{existing_dir}"
            )
        if len(epws) > 1:
            raise RuntimeError(
                f"Expected exactly one .epw in {existing_dir}, found "
                f"{len(epws)}: {[p.name for p in epws]}"
            )
        meta_expr = extract_discovery_metadata(
            Constant(epjson_path), Constant(epws[0])
        )
        meta = realize(STORE_PATH.get(), meta_expr)
        net_conditioned_area = float(meta.net_conditioned_area)
        warmup_phases = int(meta.warmup_phases)
        existing_meta = json.loads((existing_dir / "metadata.json").read_text())
        place = existing_meta.get("place", "")
    else:
        existing_meta = json.loads((existing_dir / "metadata.json").read_text())
        net_conditioned_area = float(existing_meta["net_conditioned_area"])
        warmup_phases = int(existing_meta["warmup_phases"])
        place = existing_meta.get("place", "")

    new_meta = {
        "net_conditioned_area": net_conditioned_area,
        "warmup_phases": warmup_phases,
        "source_building_id": source_building_id,
        "place": place,
    }
    with open(target_dir / "metadata.json", "w") as f:
        json.dump(new_meta, f, indent=2)

    # Count actuators for the parquet update.
    num_actuators = sum(len(e.actuator_descriptions()) for e in equipment_list)

    return {
        "building_id": building_id,
        "num_actuators": num_actuators,
        "net_conditioned_area_m2": net_conditioned_area,
        "warmup_phases": warmup_phases,
    }


def rebuild_metadata_parquet(
    out_root: Path,
    building_types_regenerated: list[BuildingType],
) -> None:
    """Rewrite ``metadata.parquet`` so that the ``action_dim`` column
    matches the new actuator count for the regenerated building types.

    Rows for non-regenerated building types are preserved bit-identically
    from the existing HF metadata.parquet so a partial regen (e.g.
    OfficeMedium only) does not perturb the other building types."""
    src_parquet = download_metadata()
    df = pd.read_parquet(src_parquet)

    for bt in building_types_regenerated:
        mask = df["building_type"] == bt
        bt_dir = out_root / bt
        for idx in df.index[mask]:
            building_id = df.at[idx, "building_id"]
            per_bldg_meta = bt_dir / building_id / "metadata.json"
            if not per_bldg_meta.exists():
                # Building was not regenerated by this Slurm shard;
                # leave the row alone so a partial shard run can still
                # write a partial parquet (re-runs of remaining shards
                # will overwrite).
                logger.warning(
                    "Skipping action_dim update for %s: %s missing",
                    building_id,
                    per_bldg_meta,
                )
                continue
            eq_path = bt_dir / building_id / "equipment.json"
            with open(eq_path) as f:
                eq = json.load(f)
            n_actuators = 0
            for entry in eq:
                if entry["equipment_type"] == "vavsystem":
                    n_actuators += 1  # SAT
                    n_actuators += 3 * len(entry["terminals"])  # flow + htg + clg
                    n_actuators += 1  # oa mass flow
                elif entry["equipment_type"] == "unitarysystem":
                    n_actuators += len(entry["actuators"])
                elif entry["equipment_type"] == "heatingonlyzone":
                    n_actuators += len(entry["actuators"])
                else:
                    raise ValueError(
                        f"Unknown equipment_type {entry['equipment_type']!r} "
                        f"in {eq_path}"
                    )
            df.at[idx, "action_dim"] = int(n_actuators)

    out_parquet = out_root / "metadata.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info("Wrote %s (%d rows)", out_parquet, len(df))


def copy_splits(out_root: Path) -> None:
    """Copy ``splits.json`` unchanged into the staging dir."""
    src = download_splits()
    dst = out_root / "splits.json"
    shutil.copy(src, dst)
    logger.info("Copied %s -> %s", src, dst)


def regen_building_type(
    building_type: BuildingType,
    out_root: Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    rerun_discovery: bool = False,
    max_per_type: int | None = None,
) -> None:
    """Regenerate every building of ``building_type`` whose split-index
    modulo ``shard_count`` equals ``shard_index``.

    The sharding is index-based (deterministic per building_id list)
    so different Slurm array tasks regenerate disjoint subsets of the
    1000 buildings.
    """
    existing_dir_root = download_building_type(building_type)
    building_ids = sorted(
        p.name for p in existing_dir_root.iterdir() if p.is_dir()
    )
    if not building_ids:
        raise RuntimeError(
            f"No buildings found in {existing_dir_root}; HF cache empty?"
        )
    if max_per_type is not None:
        building_ids = building_ids[:max_per_type]

    my_ids = [
        b
        for i, b in enumerate(building_ids)
        if i % shard_count == shard_index
    ]
    logger.info(
        "Shard %d/%d: regenerating %d/%d %s buildings",
        shard_index,
        shard_count,
        len(my_ids),
        len(building_ids),
        building_type,
    )

    t0 = time.monotonic()
    for k, bid in enumerate(my_ids, 1):
        try:
            summary = regen_one_building(
                building_type=building_type,
                building_id=bid,
                existing_dir=existing_dir_root / bid,
                out_root=out_root,
                rerun_discovery=rerun_discovery,
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "[%d/%d] %s: %d actuators (%.1fs elapsed)",
                k,
                len(my_ids),
                bid,
                summary["num_actuators"],
                elapsed,
            )
        except Exception:
            logger.exception("Failed to regenerate %s", bid)
            raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--building-type",
        action="append",
        choices=list(SUPPORTED_BUILDING_TYPES),
        required=True,
        help="Building type to regenerate. Can be repeated to regen "
        "multiple types in one run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Staging directory.  Per-building artefacts go to "
        "<output_dir>/<building_type>/<building_id>/; the unified "
        "metadata.parquet and splits.json go to <output_dir>/.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Slurm array shard index (0-based).",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of Slurm array shards.  Must be the same "
        "across all sibling array tasks.",
    )
    parser.add_argument(
        "--rerun-discovery",
        action="store_true",
        help="Re-run the EnergyPlus discovery simulation per building "
        "rather than copying net_conditioned_area / warmup_phases from "
        "the existing metadata.json.  ~10x slower; required only if the "
        "upstream pipeline has changed RunPeriod, sizing, or geometry.",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="For smoke tests: cap the number of buildings per type "
        "(applied before sharding).",
    )
    parser.add_argument(
        "--write-metadata-parquet",
        action="store_true",
        help="Also rewrite <output_dir>/metadata.parquet and copy "
        "<output_dir>/splits.json.  Only the last Slurm array task "
        "(shard_index == shard_count - 1) should pass this; the other "
        "tasks would race.  For single-shard runs, always pass this.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for bt in args.building_type:
        regen_building_type(
            building_type=bt,
            out_root=args.output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            rerun_discovery=args.rerun_discovery,
            max_per_type=args.max_per_type,
        )

    if args.write_metadata_parquet:
        rebuild_metadata_parquet(args.output_dir, args.building_type)
        copy_splits(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
