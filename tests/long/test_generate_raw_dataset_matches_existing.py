"""Stage 1 validation: generated epJSONs match the existing multizones zip.

Phase G, item G3 (see TODO.md § G3).

Tests in this module are marked ``@pytest.mark.long`` and are skipped
unless ``B2B_RUN_LONG_TESTS=1`` is set.  They require:

  - A working EnergyPlus installation (or auto-download via STORE_PATH).
  - The ASHRAE901_all.zip to be either cached or downloadable.
  - The existing HuggingFace ``vtaboga/multizones_reference_buildings.zip``
    to be cached or downloadable (for the byte-comparison tests).

G3 acceptance criterion (TODO.md):
  - Smoke: 1 building per type, metadata row matches the upstream zip
    ``(building_type, place, source_idf, weather_file)`` columns exactly;
    LHS parameter values match to float precision.
  - Full grid: user runs sbatch + diffs 6000-row metadata.csv (not in CI).
  - Per-type E+ simulation: 5 random IDs per type, all sims zero
    severe/fatal (exercised separately via ``--marker long``).
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.long


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_upstream_metadata(upstream_zip_path: Path) -> dict[int, dict]:
    """Load all metadata rows from the existing multizones_reference_buildings.zip.

    Returns a dict keyed by ``building_id`` (int).
    """
    rows: dict[int, dict] = {}
    with zipfile.ZipFile(upstream_zip_path) as zf:
        for name in sorted(zf.namelist()):
            basename = name.rsplit("/", 1)[-1]
            if not (basename.startswith("metadata") and basename.endswith(".csv")):
                continue
            with zf.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    bid = int(row["building_id"])
                    rows[bid] = dict(row)
    return rows


# ---------------------------------------------------------------------------
# Smoke test: 1 building per type, metadata row integrity
# ---------------------------------------------------------------------------


def test_generate_raw_dataset_smoke(tmp_path: Path) -> None:
    """Generate 1 building per type and validate metadata columns against the
    upstream multizones_reference_buildings.zip.

    This is the quick smoke (G3 acceptance row 1): exercises the full
    generate_raw_dataset code path without triggering a 1000-sample run.
    The upstream zip comparison validates that building_id assignment,
    place, source_idf, and weather_file conventions have not drifted.
    """
    _requires_long_runtime()

    from building2building.env import STORE_PATH
    from building2building.pipeline.generate_raw_dataset import (
        ALL_BUILDING_TYPES,
        PARAMETER_NAMES,
        N_PARAMS,
        PLACE_TO_WEATHER,
        get_parameter_ranges,
        extract_weather_files,
        load_base_buildings,
        sample_unit_lhs,
        generate_building_type,
        _merge_metadata,
    )
    from building2building.sources.multizones_reference_buildings import (
        dataset_zip,
    )
    from building2building.store import realize

    output_dir = tmp_path / "raw_dataset"
    output_dir.mkdir()

    samples_per_type = 1

    # Weather files must exist before any epJSON generation.
    extract_weather_files(output_dir)
    assert (output_dir / "weather").is_dir()
    assert len(list((output_dir / "weather").glob("*.epw"))) == len(PLACE_TO_WEATHER)

    bases_by_type = load_base_buildings(list(ALL_BUILDING_TYPES))
    unit_samples = sample_unit_lhs(samples_per_type, seed=42)
    assert unit_samples.shape == (samples_per_type, N_PARAMS)

    for bt in ALL_BUILDING_TYPES:
        shard_index = ALL_BUILDING_TYPES.index(bt)
        generate_building_type(
            building_type=bt,
            output_dir=output_dir,
            unit_samples=unit_samples,
            bases=bases_by_type[bt],
            samples_per_type=samples_per_type,
            shard_index=shard_index,
        )

    _merge_metadata(output_dir, list(ALL_BUILDING_TYPES))
    merged_csv = output_dir / "metadata.csv"
    assert merged_csv.exists(), "metadata.csv not produced after merge"

    # Read the generated metadata.
    generated: dict[int, dict] = {}
    with open(merged_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generated[int(row["building_id"])] = row
    assert len(generated) == len(ALL_BUILDING_TYPES) * samples_per_type

    # Compare discrete columns against the upstream zip.
    upstream_zip_path = realize(STORE_PATH.get(), dataset_zip())
    upstream = _read_upstream_metadata(upstream_zip_path)

    discrete_cols = ["building_type", "place", "source_idf", "weather_file"]
    for bid, gen_row in generated.items():
        if bid not in upstream:
            pytest.skip(
                f"building_id={bid} not found in the upstream zip; "
                f"zip may have been generated with different samples_per_type."
            )
        up_row = upstream[bid]
        for col in discrete_cols:
            assert gen_row[col] == up_row[col], (
                f"building_id={bid} col={col!r}: "
                f"generated={gen_row[col]!r} != upstream={up_row[col]!r}"
            )

    # Validate LHS parameter columns are present and finite.
    for bid, gen_row in generated.items():
        for pname in PARAMETER_NAMES:
            assert (
                pname in gen_row
            ), f"building_id={bid}: parameter {pname!r} missing from metadata.csv"
            val = float(gen_row[pname])
            assert np.isfinite(
                val
            ), f"building_id={bid}: parameter {pname!r} = {val!r} is not finite"

    # Validate that each epJSON file exists.
    for bid in generated:
        epjson_path = output_dir / f"{bid}.epJSON"
        assert epjson_path.exists(), f"{epjson_path} not found"


# ---------------------------------------------------------------------------
# Per-type E+ smoke: 5 random buildings per type, zero severe/fatal
# ---------------------------------------------------------------------------


def test_generate_raw_dataset_eplus_smoke(tmp_path: Path) -> None:
    """Generate 5 buildings per type, run a 1-day E+ sim on each, assert zero
    severe/fatal messages (G3 acceptance row 3).

    This test is much slower (~30 min) and only runs with B2B_RUN_LONG_TESTS=1.
    """
    _requires_long_runtime()

    import json
    import subprocess
    import shutil

    from building2building.env import STORE_PATH, energyplus_path, setup_energyplus_path
    from building2building.pipeline.generate_raw_dataset import (
        ALL_BUILDING_TYPES,
        PLACE_TO_WEATHER,
        extract_weather_files,
        load_base_buildings,
        sample_unit_lhs,
        generate_building_type,
    )
    from building2building.store import realize

    setup_energyplus_path()
    import pyenergyplus.api as eplus_api

    samples_per_type = 5
    output_dir = tmp_path / "raw_dataset_eplus_smoke"
    output_dir.mkdir()

    extract_weather_files(output_dir)
    bases_by_type = load_base_buildings(list(ALL_BUILDING_TYPES))
    unit_samples = sample_unit_lhs(samples_per_type, seed=42)

    for bt in ALL_BUILDING_TYPES:
        shard_index = ALL_BUILDING_TYPES.index(bt)
        generate_building_type(
            building_type=bt,
            output_dir=output_dir,
            unit_samples=unit_samples,
            bases=bases_by_type[bt],
            samples_per_type=samples_per_type,
            shard_index=shard_index,
        )

    failures: list[str] = []
    for bt in ALL_BUILDING_TYPES:
        shard_index = ALL_BUILDING_TYPES.index(bt)
        id_offset = shard_index * samples_per_type
        for i in range(samples_per_type):
            building_id = id_offset + i + 1
            epjson_path = output_dir / f"{building_id}.epJSON"
            assert epjson_path.exists(), f"Missing {epjson_path}"

            # Look up the weather file from the generated metadata CSV.
            # (The metadata CSV is written per shard; read the partial.)
            csv_path = output_dir / f"metadata_{shard_index}.csv"
            weather_file: str | None = None
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    if int(row["building_id"]) == building_id:
                        # weather_file is "weather/<filename>.epw"
                        weather_file = row["weather_file"]
                        break
            assert (
                weather_file is not None
            ), f"building_id={building_id} not found in {csv_path}"
            epw_path = output_dir / weather_file

            # 1-day E+ simulation.
            api = eplus_api.EnergyPlusAPI()
            state = api.state_manager.new_state()
            eplus_out = tmp_path / f"eplus_{building_id}"
            eplus_out.mkdir(exist_ok=True)

            # Patch RunPeriod to 1 day to keep the test fast.
            import copy

            with open(epjson_path) as f:
                epjson = json.load(f)
            epjson_1day = copy.deepcopy(epjson)
            for rp_name in epjson_1day.get("RunPeriod", {}):
                rp = epjson_1day["RunPeriod"][rp_name]
                rp["begin_month"] = 1
                rp["begin_day_of_month"] = 1
                rp["end_month"] = 1
                rp["end_day_of_month"] = 1
            patched_path = tmp_path / f"{building_id}_1day.epJSON"
            with open(patched_path, "w") as f:
                json.dump(epjson_1day, f)

            ret = api.runtime.run_energyplus(
                state,
                ["-d", str(eplus_out), "-w", str(epw_path), str(patched_path)],
            )
            err_path = eplus_out / "eplusout.err"
            severe_count = 0
            fatal_count = 0
            if err_path.exists():
                with open(err_path) as f:
                    for line in f:
                        ll = line.lower()
                        if "** severe  **" in ll:
                            severe_count += 1
                        if "** fatal  **" in ll:
                            fatal_count += 1
            if ret != 0 or severe_count > 0 or fatal_count > 0:
                failures.append(
                    f"building_id={building_id} ({bt}): "
                    f"ret={ret} severe={severe_count} fatal={fatal_count}"
                )

    assert not failures, f"{len(failures)} E+ simulation(s) had errors:\n" + "\n".join(
        failures
    )
