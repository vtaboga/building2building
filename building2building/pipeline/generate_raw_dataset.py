"""Stage 1 of the dataset generation pipeline.

Applies Latin Hypercube Sampling over 7 building parameters to the
16 ASHRAE 90.1-2022 prototype IDFs, producing the raw epJSON dataset
that serves as input to Stage 2
(``building2building/pipeline/generate_dataset.py``).

This module is the modernised replacement for an older ad-hoc
``scripts/generate_dataset.py``.  The LHS logic and parameter ranges
are carried over verbatim from that script.

Two-stage architecture overview::

    Stage 1 (this module)
        ASHRAE 90.1-2022 base IDFs (96 = 6 types × 16 climate locs)
            + Latin Hypercube Sampling over 7 envelope/geometry params
        ↓
        vtaboga/multizones_reference_buildings.zip layout
            weather/*.epw (16 files)
            1.epJSON .. 6000.epJSON
            metadata.csv (building_id, building_type, place,
                          source_idf, weather_file, + 7 LHS columns)

    Stage 2  →  building2building/pipeline/generate_dataset.py

Canonical single-machine run (all 6 types, 1000 samples each)::

    python -m building2building.pipeline.generate_raw_dataset \\
        --output-dir <staging> \\
        --merge-metadata

Slurm array (one task per building type, last task merges metadata)::

    sbatch building2building/pipeline/scripts/generate_raw_dataset.sh

This SLURM script is provided as an example to submit manually; it is
not auto-submitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats.qmc import LatinHypercube

from building2building.env import STORE_PATH, energyplus_path
from building2building.simulator.generator import (
    BuildingModification,
    apply_modifications,
    convert_to_epjson,
)
from building2building.sources.ashrae_90_1 import (
    ASHRAE901_all_zip,
    search_buildings,
    search_weathers,
)
from building2building.store import LocalFile, realize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical building type order (determines building_id assignment).
# Must stay fixed; changing this invalidates all downstream IDs.
# ---------------------------------------------------------------------------

ALL_BUILDING_TYPES: list[str] = [
    "Warehouse",
    "HotelSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

# ---------------------------------------------------------------------------
# ASHRAE climate location → EPW filename (16 climate locations)
# ---------------------------------------------------------------------------

PLACE_TO_WEATHER: dict[str, str] = {
    "Albuquerque": "USA_NM_Albuquerque.Intl.Sunport.723650_TMY3.epw",
    "Atlanta": "USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw",
    "Buffalo": "USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw",
    "Denver": "USA_CO_Denver-Aurora-Buckley.AFB.724695_TMY3.epw",
    "ElPaso": "USA_TX_El.Paso.Intl.AP.722700_TMY3.epw",
    "Fairbanks": "USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw",
    "GreatFalls": "USA_MT_Great.Falls.Intl.AP.727750_TMY3.epw",
    "InternationalFalls": "USA_MN_International.Falls.Intl.AP.727470_TMY3.epw",
    "Miami": "USA_FL_Miami.Intl.AP.722020_TMY3.epw",
    "NewYork": "USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw",
    "PortAngeles": "USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw",
    "Rochester": "USA_MN_Rochester.Intl.AP.726440_TMY3.epw",
    "SanDiego": "USA_CA_San.Deigo-Brown.Field.Muni.AP.722904_TMY3.epw",
    "Seattle": "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
    "Tampa": "USA_FL_Tampa-MacDill.AFB.747880_TMY3.epw",
    "Tucson": "USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw",
}

# The upstream ASHRAE/NREL TMY3 distribution ships the San Diego Brown Field
# file with a typo ("San.Deigo"). We locate the source file by its real name
# (PLACE_TO_WEATHER) but PUBLISH it under the corrected spelling so the
# metadata weather_file column and the per-building .epw filename always agree.
WEATHER_PUBLISH_RENAMES: dict[str, str] = {
    "USA_CA_San.Deigo-Brown.Field.Muni.AP.722904_TMY3.epw":
    "USA_CA_San.Diego-Brown.Field.Muni.AP.722904_TMY3.epw",
}


def published_weather_filename(source_name: str) -> str:
    """Map a source EPW filename to its published (corrected) filename."""
    return WEATHER_PUBLISH_RENAMES.get(source_name, source_name)


# ---------------------------------------------------------------------------
# ASHRAE 90.1-2022 climate zones (Tables 5.5-1 … 5.5-8)
# Used to compute climate-specific LHS parameter ranges.
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class ASHRAEFenestration:
    """ASHRAE 90.1-2022 prescriptive maximums for a given climate zone."""

    u_max: float
    shgc_max: float
    wall_u: float


ASHRAE_BY_CZ: dict[int, ASHRAEFenestration] = {
    1: ASHRAEFenestration(u_max=0.50, shgc_max=0.23, wall_u=0.124),
    2: ASHRAEFenestration(u_max=0.45, shgc_max=0.25, wall_u=0.084),
    3: ASHRAEFenestration(u_max=0.42, shgc_max=0.25, wall_u=0.077),
    4: ASHRAEFenestration(u_max=0.36, shgc_max=0.36, wall_u=0.064),
    5: ASHRAEFenestration(u_max=0.36, shgc_max=0.38, wall_u=0.055),
    6: ASHRAEFenestration(u_max=0.34, shgc_max=0.38, wall_u=0.049),
    7: ASHRAEFenestration(u_max=0.29, shgc_max=0.40, wall_u=0.049),
    8: ASHRAEFenestration(u_max=0.26, shgc_max=0.40, wall_u=0.037),
}

# Reference wall U-factor (CZ 1) for the envelope conductivity scale upper bound.
_CZ1_WALL_U = ASHRAE_BY_CZ[1].wall_u

# ---------------------------------------------------------------------------
# LHS parameter definitions
# ---------------------------------------------------------------------------

# The 7 LHS parameter names, in the exact order used by sample_unit_lhs.
# Matches the columns written to metadata.csv.
PARAMETER_NAMES: list[str] = [
    "envelope_conductivity_scale",
    "window_u_factor",
    "window_shgc",
    "infiltration_scale",
    "north_axis",
    "scale_x",
    "scale_y",
]

N_PARAMS: int = len(PARAMETER_NAMES)


@dataclass(frozen=True)
class ParameterRange:
    name: str
    low: float
    high: float


def get_parameter_ranges(climate_zone: int) -> list[ParameterRange]:
    """Return climate-dependent parameter ranges for all 7 building parameters.

    Lifted verbatim from ``scripts/generate_dataset.py`` at commit ``def97d8``.
    The ranges are calibrated against ASHRAE 90.1-2022 prescriptive maxima so
    that sampled buildings span the code-compliant envelope space for each
    climate zone.
    """
    if climate_zone not in ASHRAE_BY_CZ:
        raise ValueError(
            f"Unknown climate zone {climate_zone}; "
            f"valid: {sorted(ASHRAE_BY_CZ.keys())}"
        )
    a = ASHRAE_BY_CZ[climate_zone]

    envelope_max = min(2.0, a.wall_u / _CZ1_WALL_U * 2.0)

    if climate_zone <= 3:
        infiltration_max = 2.0
    elif climate_zone <= 5:
        infiltration_max = 1.5
    else:
        infiltration_max = 1.2

    shgc_high = min(0.80, a.shgc_max + 0.10)
    shgc_low = 0.15

    return [
        ParameterRange("envelope_conductivity_scale", 0.5, round(envelope_max, 3)),
        ParameterRange(
            "window_u_factor", round(0.8 * a.u_max, 3), round(1.3 * a.u_max, 3)
        ),
        ParameterRange("window_shgc", shgc_low, round(shgc_high, 3)),
        ParameterRange("infiltration_scale", 0.5, infiltration_max),
        ParameterRange("north_axis", 0.0, 360.0),
        ParameterRange("scale_x", 0.7, 1.5),
        ParameterRange("scale_y", 0.7, 1.5),
    ]


def sample_unit_lhs(n_samples: int, seed: int) -> np.ndarray:
    """Return an ``(n_samples × N_PARAMS)`` array of LHS unit samples in [0, 1].

    The returned matrix is deterministic: same ``n_samples`` + ``seed``
    always produces the same matrix.  The climate-specific scaling is applied
    separately (per building) in :func:`unit_to_modification`.
    """
    sampler = LatinHypercube(d=N_PARAMS, seed=seed)
    return sampler.random(n=n_samples)


def unit_to_modification(
    unit_row: np.ndarray, ranges: list[ParameterRange]
) -> BuildingModification:
    """Map a single [0, 1] unit LHS row to a BuildingModification via *ranges*."""
    if len(unit_row) != len(ranges):
        raise ValueError(
            f"unit_row length {len(unit_row)} != len(ranges) {len(ranges)}"
        )
    kwargs: dict[str, float] = {}
    for j, pr in enumerate(ranges):
        kwargs[pr.name] = float(pr.low + unit_row[j] * (pr.high - pr.low))
    return BuildingModification(**kwargs)


# ---------------------------------------------------------------------------
# Base building loading
# ---------------------------------------------------------------------------


@dataclass
class BaseBuilding:
    building_type: str
    place: str
    weather_file: str
    source_idf: str
    epjson: dict[str, Any]


def load_base_buildings(
    building_types: list[str],
) -> dict[str, list[BaseBuilding]]:
    """Load all 2022-vintage base buildings for each type × place.

    For each (building_type, place) combination that has a matching
    weather file, converts the prototype IDF to epJSON via the store
    and loads it into memory.

    Returns a dict mapping building_type → list of BaseBuilding.
    """
    store = STORE_PATH.get()
    ep = energyplus_path()

    bases_by_type: dict[str, list[BaseBuilding]] = {bt: [] for bt in building_types}

    for btype in building_types:
        df = search_buildings(building_type=btype, year=2022)
        if df.empty:
            raise RuntimeError(
                f"No ASHRAE 90.1-2022 IDF found for building_type={btype!r}. "
                f"Has ASHRAE901_all.zip been downloaded?"
            )
        for _, row in df.iterrows():
            place = str(row["place"])
            if place not in PLACE_TO_WEATHER:
                continue
            idf_path = Path(str(row["path"]))
            source_idf = idf_path.name
            idf_der = LocalFile(idf_path)
            converted = convert_to_epjson(idf_der, ep, src_version="22.1.0")
            epjson_path = realize(store, converted)
            with open(epjson_path) as f:
                epjson_obj = json.load(f)
            bases_by_type[btype].append(
                BaseBuilding(
                    building_type=btype,
                    place=place,
                    weather_file=published_weather_filename(PLACE_TO_WEATHER[place]),
                    source_idf=source_idf,
                    epjson=epjson_obj,
                )
            )
        n = len(bases_by_type[btype])
        logger.info("  %s: %d base buildings loaded", btype, n)
        if n == 0:
            raise RuntimeError(
                f"No base buildings with a matching PLACE_TO_WEATHER entry "
                f"found for {btype!r}. "
                f"Check that PLACE_TO_WEATHER covers the actual place names."
            )

    return bases_by_type


def extract_weather_files(output_dir: Path) -> None:
    """Copy the 16 climate-location EPW files into ``<output_dir>/weather/``."""
    weather_dir = output_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    for epw_filename in PLACE_TO_WEATHER.values():
        dst = weather_dir / published_weather_filename(epw_filename)
        if dst.exists():
            continue
        rows = search_weathers(filename=epw_filename)  # locate the REAL source (Deigo)
        if rows.empty:
            raise FileNotFoundError(
                f"EPW file {epw_filename!r} not found in ASHRAE901_all "
                f"extracted tree. "
                f"Run ASHRAE901_all() to verify the zip contents."
            )
        shutil.copy(rows.iloc[0]["path"], dst)  # publish under corrected name
        logger.debug("  copied %s -> %s", epw_filename, dst.name)

    logger.info("Weather files ready at %s", weather_dir)


# ---------------------------------------------------------------------------
# Metadata CSV
# ---------------------------------------------------------------------------


def _metadata_fieldnames() -> list[str]:
    return [
        "building_id",
        "building_type",
        "place",
        "source_idf",
        "weather_file",
        *PARAMETER_NAMES,
    ]


def _merge_metadata(output_dir: Path, building_types: list[str]) -> None:
    """Concatenate ``metadata_{i}.csv`` partials into ``metadata.csv``.

    One partial file per building type, named by the building type's
    index in ``ALL_BUILDING_TYPES`` (= the shard index in a Slurm run).
    Partial files are deleted after a successful merge.
    """
    fieldnames = _metadata_fieldnames()
    rows: list[dict[str, Any]] = []
    for bt in building_types:
        shard_index = ALL_BUILDING_TYPES.index(bt)
        partial = output_dir / f"metadata_{shard_index}.csv"
        if not partial.exists():
            raise FileNotFoundError(
                f"Partial metadata file {partial} not found; "
                f"shard {shard_index} ({bt}) may not have finished."
            )
        with open(partial, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["building_id"] = int(r["building_id"])
                rows.append(r)
        logger.info("  read %s (%d rows so far)", partial.name, len(rows))

    rows.sort(key=lambda r: int(r["building_id"]))

    merged = output_dir / "metadata.csv"
    with open(merged, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Merged %d rows into %s", len(rows), merged)

    for bt in building_types:
        shard_index = ALL_BUILDING_TYPES.index(bt)
        partial = output_dir / f"metadata_{shard_index}.csv"
        if partial.exists():
            partial.unlink()
    logger.info("Removed partial metadata files")


# ---------------------------------------------------------------------------
# Per-type generation
# ---------------------------------------------------------------------------


def generate_building_type(
    building_type: str,
    output_dir: Path,
    unit_samples: np.ndarray,
    bases: list[BaseBuilding],
    samples_per_type: int,
    shard_index: int,
    *,
    force: bool = False,
) -> None:
    """Generate ``samples_per_type`` epJSON variants for one building type.

    The ``shard_index`` is the building type's position in
    ``ALL_BUILDING_TYPES``.  Building IDs start at
    ``shard_index * samples_per_type + 1`` so that different types write
    to disjoint ID ranges.

    The partial metadata is written to
    ``<output_dir>/metadata_{shard_index}.csv``.

    Args:
        building_type: ASHRAE building type label (e.g. ``"OfficeMedium"``).
        output_dir: Root output directory for all building types.
        unit_samples: ``(samples_per_type × N_PARAMS)`` unit-LHS matrix.
        bases: Pre-loaded base buildings for this type.
        samples_per_type: Number of LHS samples per type.
        shard_index: This type's index in ``ALL_BUILDING_TYPES``; sets the
            building ID offset and the partial CSV filename.
        force: If ``False`` (default), skip epJSON files that already exist.
    """
    id_offset = shard_index * samples_per_type
    n_bases = len(bases)

    # Shuffle base indices with a per-type RNG state, mirroring the
    # historical script exactly (seed + type_index advance).
    # Per commit def97d8: rng is seeded once, shuffled once per preceding type.
    #
    # IMPORTANT REPRODUCIBILITY NOTE.  This shuffle's output depends on
    # ``samples_per_type``: the `rng.shuffle(np.arange(samples_per_type))`
    # advance-loop consumes a different number of RNG draws for each
    # value of ``samples_per_type``, which then shifts the
    # ``base_indices`` permutation below.  Consequence: the
    # ``building_id -> (source_idf, place)`` mapping is only stable at
    # the canonical ``samples_per_type=1000`` used for the upstream
    # ``multizones_reference_buildings.zip``.  Tests that exercise
    # smaller ``samples_per_type`` values must therefore NOT join
    # against the upstream zip on ``building_id``; join on
    # ``(building_type, source_idf)`` instead (see
    # ``tests/long/test_generate_raw_dataset_matches_existing.py``).
    seed = 42
    rng = np.random.default_rng(seed)
    for _ in range(shard_index):
        rng.shuffle(np.arange(samples_per_type))
    base_indices = np.arange(samples_per_type) % n_bases
    rng.shuffle(base_indices)

    fieldnames = _metadata_fieldnames()
    csv_path = output_dir / f"metadata_{shard_index}.csv"

    t0 = time.monotonic()
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, unit_row in enumerate(unit_samples):
            building_id = id_offset + i + 1
            base = bases[base_indices[i]]

            epjson_path = output_dir / f"{building_id}.epJSON"
            if epjson_path.exists() and not force:
                # Still need to write the metadata row even for skipped files.
                cz = PLACE_TO_CLIMATE_ZONE[base.place]
                ranges = get_parameter_ranges(cz)
                mod = unit_to_modification(unit_row, ranges)
                row: dict[str, Any] = {
                    "building_id": building_id,
                    "building_type": building_type,
                    "place": base.place,
                    "source_idf": base.source_idf,
                    "weather_file": f"weather/{base.weather_file}",
                }
                for pname in PARAMETER_NAMES:
                    row[pname] = getattr(mod, pname)
                writer.writerow(row)
                continue

            cz = PLACE_TO_CLIMATE_ZONE[base.place]
            ranges = get_parameter_ranges(cz)
            mod = unit_to_modification(unit_row, ranges)

            epjson_obj = deepcopy(base.epjson)
            apply_modifications(epjson_obj, mod)

            with open(epjson_path, "w") as f:
                json.dump(epjson_obj, f, indent=2)

            row = {
                "building_id": building_id,
                "building_type": building_type,
                "place": base.place,
                "source_idf": base.source_idf,
                "weather_file": f"weather/{base.weather_file}",
            }
            for pname in PARAMETER_NAMES:
                row[pname] = getattr(mod, pname)
            writer.writerow(row)

            if (i + 1) % 100 == 0 or (i + 1) == samples_per_type:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  [%d/%d] building_id=%d  (%.1fs elapsed)",
                    i + 1,
                    samples_per_type,
                    building_id,
                    elapsed,
                )

    logger.info(
        "%s done: %d epJSONs → %s",
        building_type,
        samples_per_type,
        csv_path.name,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root output directory.  epJSONs go here; weather/ sub-dir holds EPWs.",
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=1000,
        help="LHS samples per building type (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LHS (default: 42).",
    )
    parser.add_argument(
        "--building-type",
        dest="building_types",
        action="append",
        choices=ALL_BUILDING_TYPES,
        help=(
            "Building type to generate.  Can be repeated.  " "Defaults to all 6 types."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help=(
            "Slurm array task index (0-based).  When set, only the "
            "building type at position shard_index in the requested "
            "building_types list is processed."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help=(
            "Total number of Slurm array shards.  Must equal the number "
            "of requested building types."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing epJSON files (default: skip existing).",
    )
    parser.add_argument(
        "--merge-metadata",
        action="store_true",
        help=(
            "After generating, merge all per-type partial metadata CSVs "
            "into a single metadata.csv.  Pass this flag from the last "
            "Slurm shard after all per-type shards finish, or on single-"
            "machine runs."
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

    building_types: list[str] = args.building_types or list(ALL_BUILDING_TYPES)

    shard_index: int | None = args.shard_index
    shard_count: int | None = args.shard_count

    # Validate sharding args.
    if (shard_index is None) != (shard_count is None):
        raise ValueError("--shard-index and --shard-count must be given together.")
    if shard_index is not None and shard_count is not None:
        if shard_count != len(building_types):
            raise ValueError(
                f"--shard-count ({shard_count}) must equal the number of "
                f"requested building types ({len(building_types)})."
            )
        if not (0 <= shard_index < shard_count):
            raise ValueError(
                f"--shard-index ({shard_index}) out of range "
                f"[0, {shard_count - 1}]."
            )
        # In sharded mode, process only the type at shard_index.
        types_to_process = [building_types[shard_index]]
    else:
        types_to_process = building_types

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Stage 1 generator: %d type(s), %d samples/type, seed=%d",
        len(types_to_process),
        args.samples_per_type,
        args.seed,
    )

    logger.info("Extracting weather files ...")
    extract_weather_files(args.output_dir)

    logger.info("Loading 2022-vintage base buildings ...")
    bases_by_type = load_base_buildings(types_to_process)

    logger.info(
        "Sampling %d unit LHS vectors (seed=%d, d=%d) ...",
        args.samples_per_type,
        args.seed,
        N_PARAMS,
    )
    unit_samples = sample_unit_lhs(args.samples_per_type, args.seed)

    for bt in types_to_process:
        type_shard_index = ALL_BUILDING_TYPES.index(bt)
        logger.info(
            "Generating %d variants for %s (shard_index=%d, id_offset=%d) ...",
            args.samples_per_type,
            bt,
            type_shard_index,
            type_shard_index * args.samples_per_type,
        )
        generate_building_type(
            building_type=bt,
            output_dir=args.output_dir,
            unit_samples=unit_samples,
            bases=bases_by_type[bt],
            samples_per_type=args.samples_per_type,
            shard_index=type_shard_index,
            force=args.force,
        )

    if args.merge_metadata:
        logger.info("Merging partial metadata CSVs ...")
        # Merge all requested types, not just the ones processed in this run
        # (other shards may have already written their partials).
        _merge_metadata(args.output_dir, building_types)
        logger.info("Done.  Output at %s", args.output_dir)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
