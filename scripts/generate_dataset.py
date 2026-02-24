#!/usr/bin/env python3
"""Generate a dataset of parametrically varied EnergyPlus epJSON files.

Uses Latin Hypercube Sampling (as in BTAP) to sample building parameter
combinations.  For each of the 6 supported building types, the LHS samples
are distributed across the 16 available climate locations (ASHRAE 90.1 2022
vintage), giving diverse base envelopes and HVAC sizing *before* the
parametric modifications are applied.

Output structure::

    dataset/
        metadata.csv
        weather/
            USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw
            ...
        1.epJSON
        2.epJSON
        ...
        6000.epJSON

Usage (all types, sequential)::

    PYTHONPATH=. python scripts/generate_dataset.py [--output dataset] \\
        [--samples-per-type 1000] [--seed 42]

Usage (single type, for SLURM array jobs)::

    PYTHONPATH=. python scripts/generate_dataset.py --type-index 3 \\
        [--output dataset] [--samples-per-type 1000] [--seed 42]

When ``--type-index`` is given, only that building type is processed and
a partial metadata file ``metadata_<type-index>.csv`` is written.  A
separate merge step (see ``scripts/slurm_generate.sh``) concatenates
the partials into the final ``metadata.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats.qmc import LatinHypercube

from b2b.env import STORE_PATH, energyplus_path
from b2b.simulator.generator import (
    BuildingModification,
    apply_modifications,
    convert_to_epjson,
)
from b2b.sources.energycodes import (
    ASHRAE901_all_zip,
    BuildingType,
    search_buildings,
    search_weathers,
)
from b2b.store import ExtractFromZip, realize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


BUILDING_TYPES: list[BuildingType] = [
    "Warehouse",
    "HotelSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

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

PLACES = sorted(PLACE_TO_WEATHER.keys())
N_PLACES = len(PLACES)


@dataclass
class ParameterRange:
    name: str
    low: float
    high: float


PARAMETER_RANGES: list[ParameterRange] = [
    ParameterRange("envelope_conductivity_scale", 0.5, 2.0),
    ParameterRange("window_u_factor", 0.8, 5.0),
    ParameterRange("window_shgc", 0.1, 0.8),
    ParameterRange("infiltration_scale", 0.5, 2.0),
    ParameterRange("north_axis", 0.0, 360.0),
    ParameterRange("scale_x", 0.5, 2.0),
    ParameterRange("scale_y", 0.5, 2.0),
]

N_PARAMS = len(PARAMETER_RANGES)


def sample_modifications(
    n_samples: int, seed: int
) -> list[BuildingModification]:
    """Generate *n_samples* BuildingModification instances via LHS."""
    sampler = LatinHypercube(d=N_PARAMS, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    modifications: list[BuildingModification] = []
    for row in unit_samples:
        kwargs: dict[str, float] = {}
        for j, pr in enumerate(PARAMETER_RANGES):
            kwargs[pr.name] = float(pr.low + row[j] * (pr.high - pr.low))
        modifications.append(BuildingModification(**kwargs))
    return modifications


@dataclass
class BaseBuilding:
    building_type: BuildingType
    place: str
    weather_file: str
    source_idf: str
    epjson: dict[str, Any]


def load_base_buildings(
    building_types: list[BuildingType] | None = None,
) -> list[BaseBuilding]:
    """Load all 2022-vintage base buildings for each type × place with a
    matching weather file. Returns a flat list."""
    store = STORE_PATH.get()
    zip_der = ASHRAE901_all_zip()
    ep = energyplus_path()

    if building_types is None:
        building_types = list(BUILDING_TYPES)

    bases: list[BaseBuilding] = []
    for btype in building_types:
        buildings_df = search_buildings(building_type=btype, year=2022)
        for _, row in buildings_df.iterrows():
            place = str(row["place"])
            if place not in PLACE_TO_WEATHER:
                continue
            idf_filename = str(row["filename"])
            idf_der = ExtractFromZip(zip_der, idf_filename)
            converted = convert_to_epjson(idf_der, ep, src_version="22.1.0")
            epjson_path = realize(store, converted)
            with open(epjson_path, "r") as f:
                epjson_obj = json.load(f)
            bases.append(
                BaseBuilding(
                    building_type=btype,
                    place=place,
                    weather_file=PLACE_TO_WEATHER[place],
                    source_idf=idf_filename,
                    epjson=epjson_obj,
                )
            )
        log.info("  %s: %d base buildings loaded", btype, sum(1 for b in bases if b.building_type == btype))
    return bases


def extract_weather_files(output_dir: Path) -> None:
    """Extract each unique EPW file into *output_dir*/weather/."""
    weather_dir = output_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    store = STORE_PATH.get()
    zip_der = ASHRAE901_all_zip()
    weathers_df = search_weathers()

    for _, row in weathers_df.iterrows():
        epw_filename = str(row["filename"])
        dst = weather_dir / epw_filename
        if dst.exists():
            continue
        epw_path = realize(store, ExtractFromZip(zip_der, epw_filename))
        shutil.copy(epw_path, dst)
        log.info("  copied %s", epw_filename)


def _metadata_fieldnames() -> list[str]:
    return [
        "building_id",
        "building_type",
        "place",
        "source_idf",
        "weather_file",
        *(pr.name for pr in PARAMETER_RANGES),
    ]


def _merge_metadata(output_dir: Path, n_partials: int) -> None:
    """Concatenate ``metadata_0.csv`` .. ``metadata_{n-1}.csv`` into
    ``metadata.csv``, sorted by building_id."""
    fieldnames = _metadata_fieldnames()
    rows: list[dict[str, Any]] = []
    for idx in range(n_partials):
        partial = output_dir / f"metadata_{idx}.csv"
        if not partial.exists():
            log.warning("Missing %s – skipping", partial)
            continue
        with open(partial, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["building_id"] = int(r["building_id"])
                rows.append(r)
        log.info("  read %s", partial.name)

    rows.sort(key=lambda r: r["building_id"])

    merged = output_dir / "metadata.csv"
    with open(merged, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Merged %d rows into %s", len(rows), merged)

    for idx in range(n_partials):
        partial = output_dir / f"metadata_{idx}.csv"
        if partial.exists():
            partial.unlink()
    log.info("Removed partial metadata files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("dataset"),
        help="Output directory (default: dataset)",
    )
    parser.add_argument(
        "--samples-per-type", type=int, default=1000,
        help="Number of LHS samples per building type (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for LHS (default: 42)",
    )
    parser.add_argument(
        "--type-index", type=int, default=None,
        help="If set, process only BUILDING_TYPES[type-index] (0..5). "
             "Writes a partial metadata_<idx>.csv instead of metadata.csv.",
    )
    parser.add_argument(
        "--merge-metadata", action="store_true",
        help="Merge all metadata_*.csv partials into metadata.csv and exit.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    samples_per_type: int = args.samples_per_type
    seed: int = args.seed
    type_index: int | None = args.type_index

    if args.merge_metadata:
        _merge_metadata(output_dir, len(BUILDING_TYPES))
        return

    if type_index is not None:
        if not 0 <= type_index < len(BUILDING_TYPES):
            parser.error(
                f"--type-index must be 0..{len(BUILDING_TYPES) - 1}, "
                f"got {type_index}"
            )
        types_to_process = [BUILDING_TYPES[type_index]]
        id_offset = type_index * samples_per_type
    else:
        types_to_process = list(BUILDING_TYPES)
        id_offset = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    n_types = len(types_to_process)
    total = samples_per_type * n_types
    log.info(
        "Generating %d variants × %d type(s) = %d buildings "
        "(across %d locations)",
        samples_per_type, n_types, total, N_PLACES,
    )

    log.info("Extracting weather files ...")
    extract_weather_files(output_dir)

    log.info("Loading 2022-vintage base buildings ...")
    all_bases = load_base_buildings(types_to_process)
    bases_by_type: dict[BuildingType, list[BaseBuilding]] = {}
    for b in all_bases:
        bases_by_type.setdefault(b.building_type, []).append(b)

    log.info("Sampling %d parameter vectors via LHS (seed=%d) ...", samples_per_type, seed)
    modifications = sample_modifications(samples_per_type, seed)

    if type_index is not None:
        csv_path = output_dir / f"metadata_{type_index}.csv"
    else:
        csv_path = output_dir / "metadata.csv"

    fieldnames = _metadata_fieldnames()

    rng = np.random.default_rng(seed)
    if type_index is not None:
        for _ in range(type_index):
            rng.shuffle(np.arange(samples_per_type))

    building_id = id_offset

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for btype in types_to_process:
            type_bases = bases_by_type[btype]
            n_bases = len(type_bases)
            log.info(
                "Generating %d variants for %s (%d base locations) ...",
                samples_per_type, btype, n_bases,
            )

            base_indices = np.arange(samples_per_type) % n_bases
            rng.shuffle(base_indices)

            for i, mod in enumerate(modifications):
                building_id += 1
                base = type_bases[base_indices[i]]

                epjson_obj = deepcopy(base.epjson)
                apply_modifications(epjson_obj, mod)

                epjson_path = output_dir / f"{building_id}.epJSON"
                with open(epjson_path, "w") as f:
                    json.dump(epjson_obj, f, indent=2)

                row: dict[str, Any] = {
                    "building_id": building_id,
                    "building_type": btype,
                    "place": base.place,
                    "source_idf": base.source_idf,
                    "weather_file": f"weather/{base.weather_file}",
                }
                for pr in PARAMETER_RANGES:
                    row[pr.name] = getattr(mod, pr.name)
                writer.writerow(row)

                if building_id % 100 == 0:
                    log.info("  %d / %d files written", building_id - id_offset, total)

    log.info("Done. %d epJSON files written to %s (%s)", total, output_dir, csv_path.name)


if __name__ == "__main__":
    main()
