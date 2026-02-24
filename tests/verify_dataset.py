#!/usr/bin/env python3
"""Verify generated dataset by running 1-day EnergyPlus simulations.

Samples 10 epJSON files per building type from the dataset metadata,
patches each to a 1-day RunPeriod (Jan 1), runs EnergyPlus, and checks
the resulting ``eplusout.err`` for fatal or severe errors.

Also runs the pipeline's ``make_all_equipment`` on each building to
discover controllable HVAC equipment and report the actuator action space.

Usage::

    PYTHONPATH=. python tests/verify_dataset.py [--dataset dataset] [--samples 10] \\
        [--types OfficeSmall OfficeMedium]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from b2b.env import STORE_PATH, energyplus_path
from b2b.pipeline.actuators import AnyEquipment, make_all_equipment
from b2b.simulator.generator import convert_to_epjson
from b2b.sources.energycodes import ASHRAE901_all_zip
from b2b.store import ExtractFromZip, realize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    building_id: int
    building_type: str
    epjson_file: str
    weather_file: str
    success: bool
    fatal_errors: list[str]
    severe_errors: int
    warnings: int
    equipment: list[AnyEquipment] = field(default_factory=list)


def patch_one_day_run_period(epjson: dict) -> dict:
    """Override every RunPeriod entry to simulate only Jan 1."""
    if "RunPeriod" not in epjson:
        raise ValueError("epJSON has no RunPeriod object")
    for key in epjson["RunPeriod"]:
        rp = epjson["RunPeriod"][key]
        rp["begin_month"] = 1
        rp["begin_day_of_month"] = 1
        rp["end_month"] = 1
        rp["end_day_of_month"] = 1
    return epjson


def parse_err_file(err_path: Path) -> tuple[bool, list[str], int, int]:
    """Parse an EnergyPlus .err file.

    Returns (success, fatal_lines, severe_count, warning_count).
    """
    if not err_path.exists():
        return False, ["eplusout.err not produced"], 0, 0

    text = err_path.read_text()
    fatal_lines = [
        line.strip()
        for line in text.splitlines()
        if "** Fatal **" in line
    ]
    severe_count = text.count("** Severe  **")
    warning_count = text.count("**   ~~~   **")
    success = len(fatal_lines) == 0 and severe_count == 0
    return success, fatal_lines, severe_count, warning_count


RTOL = 1e-6


def _all_values(epjson: dict, obj_type: str, field_name: str) -> list[float]:
    """Collect all values of *field_name* across objects of *obj_type*."""
    return [
        obj[field_name]
        for obj in epjson.get(obj_type, {}).values()
        if field_name in obj
    ]


def _first_vertex_coords(
    epjson: dict,
) -> list[tuple[float, float, float]]:
    """Collect the first vertex of every BuildingSurface:Detailed."""
    coords: list[tuple[float, float, float]] = []
    for surf in epjson.get("BuildingSurface:Detailed", {}).values():
        if "vertices" in surf and surf["vertices"]:
            v = surf["vertices"][0]
            coords.append((
                v["vertex_x_coordinate"],
                v["vertex_y_coordinate"],
                v["vertex_z_coordinate"],
            ))
        elif "vertex_1_x_coordinate" in surf:
            coords.append((
                surf["vertex_1_x_coordinate"],
                surf["vertex_1_y_coordinate"],
                surf["vertex_1_z_coordinate"],
            ))
    return coords


def verify_modifications(
    epjson_path: Path,
    metadata_row: dict[str, str],
    base_cache: dict[str, dict[str, Any]],
) -> list[str]:
    """Check that the epJSON reflects the modifications from metadata.

    Returns a list of mismatch descriptions (empty = all OK).
    """
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    errors: list[str] = []

    expected_north = float(metadata_row["north_axis"])
    for _name, bldg in epjson.get("Building", {}).items():
        actual = bldg.get("north_axis", 0.0)
        if not np.isclose(actual, expected_north, rtol=RTOL):
            errors.append(
                f"north_axis: expected {expected_north:.4f}, got {actual:.4f}"
            )

    expected_u = float(metadata_row["window_u_factor"])
    for val in _all_values(epjson, "WindowMaterial:SimpleGlazingSystem", "u_factor"):
        if not np.isclose(val, expected_u, rtol=RTOL):
            errors.append(
                f"window_u_factor: expected {expected_u:.4f}, got {val:.4f}"
            )

    expected_shgc = float(metadata_row["window_shgc"])
    for val in _all_values(
        epjson, "WindowMaterial:SimpleGlazingSystem", "solar_heat_gain_coefficient"
    ):
        if not np.isclose(val, expected_shgc, rtol=RTOL):
            errors.append(
                f"window_shgc: expected {expected_shgc:.4f}, got {val:.4f}"
            )

    source_idf = metadata_row["source_idf"]
    base = base_cache.get(source_idf)
    if base is None:
        errors.append(f"base building not loaded for {source_idf}")
        return errors

    cond_scale = float(metadata_row["envelope_conductivity_scale"])
    base_conds = _all_values(base, "Material", "conductivity")
    mod_conds = _all_values(epjson, "Material", "conductivity")
    if len(base_conds) == len(mod_conds):
        for i, (bc, mc) in enumerate(zip(base_conds, mod_conds)):
            expected = bc * cond_scale
            if not np.isclose(mc, expected, rtol=RTOL):
                errors.append(
                    f"conductivity[{i}]: base={bc:.6f} * scale={cond_scale:.4f} "
                    f"= {expected:.6f}, got {mc:.6f}"
                )
                break
    elif base_conds:
        errors.append(
            f"Material count mismatch: base={len(base_conds)}, "
            f"modified={len(mod_conds)}"
        )

    infil_scale = float(metadata_row["infiltration_scale"])
    base_infils = _all_values(
        base, "ZoneInfiltration:DesignFlowRate", "design_flow_rate"
    )
    mod_infils = _all_values(
        epjson, "ZoneInfiltration:DesignFlowRate", "design_flow_rate"
    )
    if len(base_infils) == len(mod_infils):
        for i, (bi, mi) in enumerate(zip(base_infils, mod_infils)):
            expected = bi * infil_scale
            if not np.isclose(mi, expected, rtol=RTOL):
                errors.append(
                    f"infiltration[{i}]: base={bi:.6f} * scale={infil_scale:.4f} "
                    f"= {expected:.6f}, got {mi:.6f}"
                )
                break
    elif base_infils:
        errors.append(
            f"Infiltration count mismatch: base={len(base_infils)}, "
            f"modified={len(mod_infils)}"
        )

    sx = float(metadata_row["scale_x"])
    sy = float(metadata_row["scale_y"])
    base_coords = _first_vertex_coords(base)
    mod_coords = _first_vertex_coords(epjson)
    if len(base_coords) == len(mod_coords) and base_coords:
        for i, (bc, mc) in enumerate(zip(base_coords, mod_coords)):
            ex = bc[0] * sx
            ey = bc[1] * sy
            if not (np.isclose(mc[0], ex, rtol=RTOL) and
                    np.isclose(mc[1], ey, rtol=RTOL)):
                errors.append(
                    f"geometry vertex[{i}]: base=({bc[0]:.4f},{bc[1]:.4f}) "
                    f"* ({sx:.4f},{sy:.4f}) "
                    f"= ({ex:.4f},{ey:.4f}), got ({mc[0]:.4f},{mc[1]:.4f})"
                )
                break

    return errors


def load_base_epjsons(
    source_idfs: set[str],
) -> dict[str, dict[str, Any]]:
    """Load base epJSON dicts for each unique source_idf, keyed by filename."""
    store = STORE_PATH.get()
    zip_der = ASHRAE901_all_zip()
    ep = energyplus_path()

    cache: dict[str, dict[str, Any]] = {}
    for idf_filename in sorted(source_idfs):
        idf_der = ExtractFromZip(zip_der, idf_filename)
        converted = convert_to_epjson(idf_der, ep, src_version="22.1.0")
        epjson_path = realize(store, converted)
        with open(epjson_path, "r") as f:
            cache[idf_filename] = json.load(f)
        log.info("  loaded base: %s", idf_filename)
    return cache


def discover_equipment(epjson_path: Path) -> list[AnyEquipment]:
    """Run the pipeline's make_all_equipment to discover controllable
    equipment and their actuators from a raw epJSON."""
    with open(epjson_path, "r") as f:
        epjson = json.load(f)
    _, equipment = make_all_equipment(epjson)
    return list(equipment)


def run_one_day_simulation(
    ep_dir: Path,
    epjson_path: Path,
    weather_path: Path,
) -> tuple[bool, list[str], int, int]:
    """Run a 1-day EnergyPlus simulation and return parsed error info."""
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    patch_one_day_run_period(epjson)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        patched = tmp / "in.epJSON"
        with open(patched, "w") as f:
            json.dump(epjson, f, indent=2)

        cmd = [
            str(ep_dir / "energyplus"),
            "-d", str(tmp),
            "-w", str(weather_path),
            "-x",
            str(patched),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        err_file = tmp / "eplusout.err"
        success, fatals, severe, warnings = parse_err_file(err_file)

        if result.returncode != 0 and not fatals:
            fatals.append(f"energyplus exited with code {result.returncode}")
            success = False

        return success, fatals, severe, warnings


def collect_samples(
    dataset_dir: Path,
    samples_per_type: int,
    seed: int,
    types_filter: list[str] | None = None,
) -> list[dict[str, str]]:
    """Read all metadata partials and sample rows per building type."""
    all_rows: list[dict[str, str]] = []
    for csv_path in sorted(dataset_dir.glob("metadata_*.csv")):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    if not all_rows:
        merged = dataset_dir / "metadata.csv"
        if merged.exists():
            with open(merged, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)

    if not all_rows:
        raise FileNotFoundError(
            f"No metadata CSV files found in {dataset_dir}"
        )

    by_type: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        by_type.setdefault(row["building_type"], []).append(row)

    if types_filter:
        available = set(by_type.keys())
        requested = set(types_filter)
        missing = requested - available
        if missing:
            raise ValueError(
                f"Requested types not in dataset: {missing}. "
                f"Available: {sorted(available)}"
            )
        by_type = {k: v for k, v in by_type.items() if k in requested}

    rng = np.random.default_rng(seed)
    sampled: list[dict[str, str]] = []
    for btype in sorted(by_type):
        rows = by_type[btype]
        n = min(samples_per_type, len(rows))
        indices = rng.choice(len(rows), size=n, replace=False)
        for i in indices:
            sampled.append(rows[i])
        log.info("  %s: sampled %d / %d", btype, n, len(rows))

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset"),
        help="Dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of files to sample per building type (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for sampling (default: 123)",
    )
    parser.add_argument(
        "--types", nargs="+", default=None,
        help="Building types to test (default: all). "
             "E.g. --types OfficeSmall OfficeMedium",
    )
    args = parser.parse_args()

    dataset_dir: Path = args.dataset
    if not dataset_dir.exists():
        parser.error(f"Dataset directory does not exist: {dataset_dir}")

    log.info("Resolving EnergyPlus path ...")
    store = STORE_PATH.get()
    ep_dir = realize(store, energyplus_path())
    log.info("  EnergyPlus: %s", ep_dir)

    log.info("Sampling %d buildings per type ...", args.samples)
    sampled = collect_samples(dataset_dir, args.samples, args.seed, args.types)
    log.info("Total buildings to verify: %d", len(sampled))

    log.info("Loading base buildings for modification checks ...")
    source_idfs = {row["source_idf"] for row in sampled}
    base_cache = load_base_epjsons(source_idfs)

    results: list[VerificationResult] = []
    mod_failures: list[tuple[int, str, list[str]]] = []

    for idx, row in enumerate(sampled, 1):
        bid = int(row["building_id"])
        btype = row["building_type"]
        epjson_file = dataset_dir / f"{bid}.epJSON"
        weather_file = dataset_dir / row["weather_file"]

        if not epjson_file.exists():
            log.error("[%d/%d] MISSING %s", idx, len(sampled), epjson_file)
            results.append(VerificationResult(
                building_id=bid, building_type=btype,
                epjson_file=str(epjson_file), weather_file=str(weather_file),
                success=False, fatal_errors=[f"File not found: {epjson_file}"],
                severe_errors=0, warnings=0,
            ))
            continue

        if not weather_file.exists():
            log.error("[%d/%d] MISSING weather %s", idx, len(sampled), weather_file)
            results.append(VerificationResult(
                building_id=bid, building_type=btype,
                epjson_file=str(epjson_file), weather_file=str(weather_file),
                success=False,
                fatal_errors=[f"Weather not found: {weather_file}"],
                severe_errors=0, warnings=0,
            ))
            continue

        log.info(
            "[%d/%d] Verifying %s (id=%d, %s) ...",
            idx, len(sampled), btype, bid, epjson_file.name,
        )

        mod_errors = verify_modifications(epjson_file, row, base_cache)
        if mod_errors:
            log.error("  MODIFICATION MISMATCH:")
            for e in mod_errors:
                log.error("    %s", e)
            mod_failures.append((bid, btype, mod_errors))
        else:
            log.info("  modifications OK")

        equipment = discover_equipment(epjson_file)
        if not equipment:
            log.warning("  no controllable equipment found")
        else:
            for eq in equipment:
                actuators = eq.actuator_descriptions()
                log.info(
                    "  equipment: %s  zones=%s  actuators=%d",
                    eq.equipment_type, eq.zones(), len(actuators),
                )
                for act in actuators:
                    log.info(
                        "    %s / %s  [%s]  bounds=(%.1f, %.1f)",
                        act.component_type, act.control_type,
                        act.units, act.lower_bound, act.upper_bound,
                    )

        success, fatals, severe, warnings = run_one_day_simulation(
            ep_dir, epjson_file, weather_file,
        )
        status = "OK" if success else "FAIL"
        log.info(
            "  simulation %s  severe=%d  warnings=%d%s",
            status, severe, warnings,
            f"  fatals={fatals}" if fatals else "",
        )
        results.append(VerificationResult(
            building_id=bid, building_type=btype,
            epjson_file=str(epjson_file), weather_file=str(weather_file),
            success=success, fatal_errors=fatals,
            severe_errors=severe, warnings=warnings,
            equipment=equipment,
        ))

    # --- Summary ---
    n_sim_ok = sum(1 for r in results if r.success)
    n_sim_fail = len(results) - n_sim_ok
    n_mod_fail = len(mod_failures)
    log.info("=" * 60)
    log.info(
        "SIMULATION: %d OK, %d FAILED out of %d",
        n_sim_ok, n_sim_fail, len(results),
    )
    log.info(
        "MODIFICATIONS: %d OK, %d FAILED out of %d",
        len(sampled) - n_mod_fail, n_mod_fail, len(sampled),
    )

    # Equipment / actuator summary per building type
    log.info("-" * 60)
    log.info("EQUIPMENT SUMMARY BY BUILDING TYPE:")
    by_btype: dict[str, list[VerificationResult]] = defaultdict(list)
    for r in results:
        by_btype[r.building_type].append(r)

    for btype in sorted(by_btype):
        type_results = by_btype[btype]
        eq_type_counts: dict[str, int] = defaultdict(int)
        actuator_type_set: set[tuple[str, str, str]] = set()
        total_actuators = 0
        for r in type_results:
            for eq in r.equipment:
                eq_type_counts[eq.equipment_type] += 1
                for act in eq.actuator_descriptions():
                    actuator_type_set.add(
                        (act.component_type, act.control_type, act.units)
                    )
                    total_actuators += 1

        log.info("  %s (%d buildings):", btype, len(type_results))
        for eq_type, count in sorted(eq_type_counts.items()):
            log.info("    equipment: %-20s  count=%d", eq_type, count)
        avg = total_actuators / len(type_results) if type_results else 0
        log.info("    total actuators: %d  (avg %.1f per building)", total_actuators, avg)
        log.info("    actuator types:")
        for comp_type, ctrl_type, units in sorted(actuator_type_set):
            log.info("      %s / %s  [%s]", comp_type, ctrl_type, units)

    # --- Failure report ---
    failed = False
    if n_sim_fail > 0:
        log.info("-" * 60)
        log.info("Simulation failures:")
        for r in results:
            if not r.success:
                log.info(
                    "  id=%-6d type=%-20s severe=%d fatals=%s",
                    r.building_id, r.building_type,
                    r.severe_errors, r.fatal_errors,
                )
        failed = True

    if n_mod_fail > 0:
        log.info("-" * 60)
        log.info("Modification mismatches:")
        for bid, btype, errs in mod_failures:
            log.info("  id=%-6d type=%-20s", bid, btype)
            for e in errs:
                log.info("    %s", e)
        failed = True

    if failed:
        raise SystemExit(1)
    else:
        log.info("All checks passed.")


if __name__ == "__main__":
    main()
