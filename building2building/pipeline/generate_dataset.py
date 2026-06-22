"""Stage 2 of the dataset generation pipeline.

Reads the raw epJSON archive (``vtaboga/multizones_reference_buildings``),
runs the current B2B controllability pipeline on each building, and writes
the processed artefacts consumed by training and evaluation:

  - ``<output_dir>/<BuildingType>/<BuildingType-N>/building.epjson``
  - ``<output_dir>/<BuildingType>/<BuildingType-N>/equipment.json``
  - ``<output_dir>/<BuildingType>/<BuildingType-N>/metadata.json``
  - ``<output_dir>/metadata.parquet``  (rewritten by ``--write-metadata-parquet``)
  - ``<output_dir>/splits.json``       (copied unchanged from HF)

This module replaces the older ``regen_dataset.py`` and
``regen_officemedium.sh`` helpers: run this script with
``--building-type OfficeMedium`` instead of the old
``regen_officemedium.sh``.

Key design differences from ``regen_dataset.py``:

  1. Building IDs come from ``splits.json`` (HF download) rather than
     from the locally extracted HF per-building-type zip.  Single source
     of truth, no ``~/.cache/building2building/<BT>/*.zip`` needed.

  2. ``source_building_id`` is parsed directly from the processed ID string
     (``"OfficeMedium-4001"`` → ``4001``) without a round-trip through the
     existing ``metadata.json``.  Fails loud on malformed IDs.

  3. Discovery metadata (``net_conditioned_area``, ``warmup_phases``) is
     always recomputed via a 1-day E+ simulation; there is no
     ``--rerun-discovery`` shortcut because this script does not rely on a
     pre-existing HF artefact cache.

  4. ``action_dim`` in ``metadata.parquet`` is counted directly from
     ``equipment_list`` via ``sum(len(e.actuator_descriptions()) for e in
     equipment_list)``, removing the hardcoded per-type counter in
     ``regen_dataset.py::rebuild_metadata_parquet``.

  5. Missing per-building artefact directories are a hard error (no silent
     skip); fail loudly.

Canonical single-machine run::

    python -m building2building.pipeline.generate_dataset \\
        --building-type OfficeMedium \\
        --output-dir <staging> \\
        --write-metadata-parquet

Slurm array (generic over building type)::

    sbatch building2building/pipeline/scripts/generate_dataset.sh
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
    download_metadata,
    download_splits,
)
from building2building.env import STORE_PATH, setup_energyplus_path
from building2building.pipeline.discovery import extract_discovery_metadata
from building2building.sources.multizones_reference_buildings import (
    _build_control_derivation,
    dataset_zip,
    table_index,
)
from building2building.store import Constant, ExtractFromZip, realize

logger = logging.getLogger(__name__)


# Building types supported by Stage 2.  ``SingleFamilyHouse`` has a
# different upstream source (residential.py) and is out of scope.
SUPPORTED_BUILDING_TYPES: tuple[BuildingType, ...] = (
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
)


# ---------------------------------------------------------------------------
# Building ID helpers
# ---------------------------------------------------------------------------


def _parse_source_building_id(processed_id: str) -> int:
    """Parse the source building ID from a processed ID string.

    Example: ``"OfficeMedium-4001"`` → ``4001``.

    Raises ``ValueError`` on malformed input (no fallback).
    """
    parts = processed_id.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Malformed processed_id {processed_id!r}; "
            f"expected '<BuildingType>-<int>'."
        )
    try:
        return int(parts[1])
    except ValueError:
        raise ValueError(
            f"Malformed processed_id {processed_id!r}; "
            f"suffix {parts[1]!r} is not an integer."
        )


def _load_splits(
    building_types: list[BuildingType],
) -> dict[BuildingType, list[str]]:
    """Return the union of train ∪ test ∪ test_small processed IDs per type.

    Sourced from the registry's split view rather than the raw
    ``splits.json``, so the curated ``test_small`` subset is always
    represented: when a published ``splits.json`` omits it (e.g. after a
    dataset regeneration that did not re-upload the manifest), the registry
    derives it deterministically from ``test`` (see
    :func:`building2building.data.registry.derive_test_small_split`).
    ``test_small`` ⊆ ``test`` so the union is unchanged, but test_small stays
    first-class throughout the pipeline.

    Going through the registry (rather than reading ``splits.json`` directly)
    additionally loads ``metadata.parquet`` via ``download_metadata()`` — the
    registry needs it to derive ``test_small`` when the manifest omits it. Like
    ``download_splits()`` (already used by the previous implementation), this is
    served from the HuggingFace cache and is offline-safe once the dataset has
    been fetched, which any real generation run already does. The
    ``get_registry`` import is function-local because ``registry`` does not
    import this module — there is no import cycle, but keeping it local avoids
    one if that ever changes.

    Raises ``KeyError`` if a requested building type is absent from a split.
    """
    from building2building.data.registry import get_registry

    splits = get_registry().splits

    result: dict[BuildingType, list[str]] = {}
    for bt in building_types:
        seen: set[str] = set()
        ids: list[str] = []
        for split_name in ("train", "test", "test_small"):
            split_data = splits.get(split_name, {})
            if bt not in split_data:
                raise KeyError(
                    f"Building type {bt!r} not found in splits "
                    f"split {split_name!r}. "
                    f"Available: {sorted(split_data.keys())}"
                )
            for pid in split_data[bt]:
                if pid not in seen:
                    ids.append(pid)
                    seen.add(pid)
        result[bt] = ids
    return result


# ---------------------------------------------------------------------------
# Per-building generation
# ---------------------------------------------------------------------------


def generate_one_building(
    building_type: BuildingType,
    processed_id: str,
    out_root: Path,
) -> dict:
    """Generate controllable artefacts for one building.

    1. Parse ``source_building_id`` from ``processed_id``.
    2. Run ``_build_control_derivation`` on the raw epJSON from the zip.
    3. Run a 1-day EnergyPlus discovery simulation to extract
       ``net_conditioned_area`` and ``warmup_phases``.
    4. Write ``building.epjson``, ``equipment.json``, ``metadata.json``
       under ``out_root/<building_type>/<processed_id>/``.

    Returns a summary dict for ``metadata.parquet`` aggregation.
    """
    target_dir = out_root / building_type / processed_id
    target_dir.mkdir(parents=True, exist_ok=True)

    source_building_id = _parse_source_building_id(processed_id)
    epjson_filename = f"{source_building_id}.epJSON"

    root_zip = dataset_zip()
    derivation = _build_control_derivation(
        root_zip=root_zip,
        epjson_filename=epjson_filename,
        run_period_name="full_year",
    )
    epjson_path, equipment_list = realize(STORE_PATH.get(), derivation)

    shutil.copy(epjson_path, target_dir / "building.epjson")
    with open(target_dir / "equipment.json", "w") as f:
        json.dump(unstructure(list(equipment_list)), f, indent=4)

    # Look up the weather file from the multizones_reference_buildings metadata.
    idx_path = realize(STORE_PATH.get(), table_index(root_zip))
    import duckdb

    df_meta = (
        duckdb.from_parquet(str(idx_path))
        .filter(
            duckdb.ColumnExpression("building_id")
            == duckdb.ConstantExpression(source_building_id)
        )
        .to_df()
    )
    if df_meta.empty:
        raise RuntimeError(
            f"building_id={source_building_id} not found in "
            f"multizones_reference_buildings table index."
        )
    weather_file = str(df_meta.iloc[0]["weather_file"])
    # weather_file is like "weather/USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw"
    epw_derivation = ExtractFromZip(root_zip, weather_file)

    meta_expr = extract_discovery_metadata(Constant(epjson_path), epw_derivation)
    meta = realize(STORE_PATH.get(), meta_expr)
    net_conditioned_area = float(meta.net_conditioned_area)
    warmup_phases = int(meta.warmup_phases)

    # Copy the EPW into the per-building directory so it round-trips through
    # ``factory.py``/``api/__init__.py`` which locate it as
    # ``info.building_dir / info.weather_file`` (see
    # building2building/envs/factory.py:55 and building2building/api/__init__.py:294).
    # ``realize`` returns the store path ``<derivation_hash>-<basename>``
    # (see building2building/store.py:73), and the existing HF
    # ``metadata.parquet`` ``weather_file`` column (preserved unchanged by
    # ``rebuild_metadata_parquet``) carries exactly that same name from the
    # original Stage-2 run.  Copying with the store basename preserves the
    # filename contract end-to-end.
    epw_path = realize(STORE_PATH.get(), epw_derivation)
    shutil.copy(epw_path, target_dir / epw_path.name)

    place = str(df_meta.iloc[0].get("place", ""))

    new_meta = {
        "net_conditioned_area": net_conditioned_area,
        "warmup_phases": warmup_phases,
        "source_building_id": source_building_id,
        "place": place,
    }
    with open(target_dir / "metadata.json", "w") as f:
        json.dump(new_meta, f, indent=2)

    num_actuators = sum(len(e.actuator_descriptions()) for e in equipment_list)

    return {
        "building_id": processed_id,
        "num_actuators": num_actuators,
        "net_conditioned_area_m2": net_conditioned_area,
        "warmup_phases": warmup_phases,
    }


def generate_building_type(
    building_type: BuildingType,
    out_root: Path,
    all_ids: list[str],
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    force: bool = False,
) -> None:
    """Generate all buildings of ``building_type`` assigned to this shard.

    Sharding is index-based over ``all_ids``: shard N processes IDs whose
    position in ``all_ids`` satisfies ``position % shard_count == shard_index``.
    """
    my_ids = [pid for i, pid in enumerate(all_ids) if i % shard_count == shard_index]
    logger.info(
        "Shard %d/%d: %d/%d %s buildings to process",
        shard_index,
        shard_count,
        len(my_ids),
        len(all_ids),
        building_type,
    )

    t0 = time.monotonic()
    for k, pid in enumerate(my_ids, 1):
        target_dir = out_root / building_type / pid
        # A complete per-building dir holds the 3 JSON/epJSON artefacts plus
        # the canonical-name EPW (``<sha256>-<basename>.epw``).  Any of these
        # missing means the previous run was partial and we must regenerate.
        if (
            not force
            and target_dir.is_dir()
            and (
                (target_dir / "building.epjson").exists()
                and (target_dir / "equipment.json").exists()
                and (target_dir / "metadata.json").exists()
                and any(target_dir.glob("*.epw"))
            )
        ):
            logger.debug("[%d/%d] %s already exists, skipping.", k, len(my_ids), pid)
            continue

        summary = generate_one_building(
            building_type=building_type,
            processed_id=pid,
            out_root=out_root,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "[%d/%d] %s: %d actuators (%.1fs elapsed)",
            k,
            len(my_ids),
            pid,
            summary["num_actuators"],
            elapsed,
        )


# ---------------------------------------------------------------------------
# metadata.parquet rebuild
# ---------------------------------------------------------------------------


def rebuild_metadata_parquet(
    out_root: Path,
    building_types_regenerated: list[BuildingType],
) -> None:
    """Rewrite ``metadata.parquet`` with updated ``action_dim`` for the
    regenerated building types.

    ``action_dim`` is the **agent-facing** action-space dimension — i.e.
    ``env.action_space.shape[0]`` for the env constructed by
    :func:`building2building.api.make_env`.  We compute it by routing
    the equipment list through
    :func:`building2building.simulator.action_spaces.agent_action_dim`,
    which applies the same fixed-actuator filter
    (:func:`hvac_action_space`) used by the simulator at runtime so that
    :data:`BuildingInfo.action_dim` matches the gym action space by
    construction.  Counting raw ``actuator_descriptions()`` instead
    would produce the *full* EnergyPlus actuator vector (51 for
    OfficeMedium) rather than the agent-facing dim (36).

    Rows for non-regenerated building types are preserved bit-identically
    from the existing HF ``metadata.parquet``.

    Raises if a per-building artefact directory is missing for any
    regenerated building (no silent skip; fail loudly).
    """
    src_parquet = download_metadata()
    df = pd.read_parquet(src_parquet)

    from cattrs import structure
    from building2building.pipeline.actuators import AnyEquipment
    from building2building.simulator.action_spaces import agent_action_dim

    for bt in building_types_regenerated:
        mask = df["building_type"] == bt
        bt_dir = out_root / bt
        for idx in df.index[mask]:
            processed_id = df.at[idx, "building_id"]
            bldg_dir = bt_dir / processed_id
            if not bldg_dir.exists():
                raise FileNotFoundError(
                    f"Expected regenerated artefacts at {bldg_dir} but "
                    f"directory not found.  Run all shards before "
                    f"--write-metadata-parquet."
                )
            eq_path = bldg_dir / "equipment.json"
            if not eq_path.exists():
                raise FileNotFoundError(f"equipment.json missing at {eq_path}.")
            equipment_list = structure(
                json.loads(eq_path.read_text()), list[AnyEquipment]
            )
            df.at[idx, "action_dim"] = int(agent_action_dim(equipment_list))

    out_parquet = out_root / "metadata.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info("Wrote %s (%d rows)", out_parquet, len(df))


def copy_splits(out_root: Path) -> None:
    """Copy ``splits.json`` unchanged from the HF cache into ``out_root``."""
    src = download_splits()
    dst = out_root / "splits.json"
    shutil.copy(src, dst)
    logger.info("Copied %s → %s", src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--building-type",
        dest="building_types",
        action="append",
        choices=list(SUPPORTED_BUILDING_TYPES),
        required=True,
        help=(
            "Building type to regenerate.  Can be repeated to process "
            "multiple types in one invocation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Staging directory.  Per-building artefacts go to "
            "<output_dir>/<building_type>/<processed_id>/; "
            "metadata.parquet and splits.json go to <output_dir>/."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Slurm array shard index (0-based, default: 0).",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help=(
            "Total number of Slurm shards.  Must be the same across all "
            "sibling array tasks."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing artefacts (default: skip if all three exist).",
    )
    parser.add_argument(
        "--write-metadata-parquet",
        action="store_true",
        help=(
            "Also rewrite <output_dir>/metadata.parquet and copy "
            "splits.json.  Only the last shard (shard_index == "
            "shard_count - 1) should pass this flag; other shards "
            "would race on the parquet file."
        ),
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

    # EnergyPlus must be on the Python path for pyenergyplus to import.
    setup_energyplus_path()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ids_by_type = _load_splits(args.building_types)

    for bt in args.building_types:
        generate_building_type(
            building_type=bt,
            out_root=args.output_dir,
            all_ids=ids_by_type[bt],
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            force=args.force,
        )

    if args.write_metadata_parquet:
        rebuild_metadata_parquet(args.output_dir, args.building_types)
        copy_splits(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
