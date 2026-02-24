"""Integration tests for the parametric building generator.

Each test picks one building from the EnergyCodes dataset for a given type,
applies parametric modifications, runs a **design-day-only** EnergyPlus
simulation, and asserts that the simulation completes without fatal errors.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

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

BUILDING_TYPES: list[BuildingType] = [
    "ApartmentMidRise",
    "Warehouse",
    "ApartmentHighRise",
    "HotelSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

TEST_MODIFICATION = BuildingModification(
    envelope_conductivity_scale=1.5,
    window_u_factor=2.0,
    window_shgc=0.4,
    infiltration_scale=1.2,
    north_axis=90.0,
    scale_x=1.1,
    scale_y=0.9,
)


def _pick_building_idf(building_type: BuildingType) -> str:
    """Return the IDF filename for one building of the given type."""
    buildings = search_buildings(building_type=building_type)
    return str(buildings.iloc[0]["filename"])


def _pick_weather_file() -> str:
    """Return the filename of one weather file from the EnergyCodes archive."""
    weathers = search_weathers()
    return str(weathers.iloc[0]["filename"])


def _make_design_day_only(epjson: dict) -> None:
    """Strip RunPeriod and run only SizingPeriod:DesignDay (seconds, not
    minutes)."""
    epjson.pop("RunPeriod", None)
    sim_ctrl = epjson.setdefault("SimulationControl", {})
    for _name, ctrl in sim_ctrl.items():
        ctrl["run_simulation_for_sizing_periods"] = "Yes"
        ctrl["run_simulation_for_weather_file_run_periods"] = "No"
    if not sim_ctrl:
        sim_ctrl["SimulationControl 1"] = {
            "do_zone_sizing_calculation": "Yes",
            "do_system_sizing_calculation": "Yes",
            "do_plant_sizing_calculation": "No",
            "run_simulation_for_sizing_periods": "Yes",
            "run_simulation_for_weather_file_run_periods": "No",
        }


def _run_eplus_designday(
    ep_dir: Path, epjson_path: Path, epw_path: Path
) -> Path:
    """Write a design-day-only copy of *epjson_path*, run EnergyPlus, and
    return the output directory."""
    with open(epjson_path, "r") as f:
        epjson = json.load(f)
    _make_design_day_only(epjson)

    tmpdir = Path(tempfile.mkdtemp())
    dd_path = tmpdir / "dd_building.epjson"
    with open(dd_path, "w") as f:
        json.dump(epjson, f, indent=2)

    outdir = tmpdir / "out"
    outdir.mkdir()
    cmd = [
        str(ep_dir / "energyplus"),
        "-d", str(outdir),
        "-w", str(epw_path),
        "-x",
        str(dd_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else ""
        stdout_tail = result.stdout[-2000:] if result.stdout else ""
        raise RuntimeError(
            f"EnergyPlus failed (rc={result.returncode}) for {epjson_path.name}\n"
            f"--- stdout ---\n{stdout_tail}\n"
            f"--- stderr ---\n{stderr_tail}"
        )
    return outdir


def _has_fatal_errors(outdir: Path) -> str | None:
    """Return the first fatal-error line from eplusout.err, or None."""
    err_file = outdir / "eplusout.err"
    if not err_file.exists():
        return "eplusout.err not found"
    for line in err_file.read_text().splitlines():
        if "** Fatal **" in line:
            return line
    return None


def _convert_one(building_type: BuildingType) -> Path:
    """Upgrade + convert one building to raw epJSON (no pipeline processing).
    Results are cached by the store."""
    store = STORE_PATH.get()
    zip_der = ASHRAE901_all_zip()
    ep = energyplus_path()
    idf_filename = _pick_building_idf(building_type)
    idf_der = ExtractFromZip(zip_der, idf_filename)
    converted = convert_to_epjson(idf_der, ep, src_version="22.1.0")
    return realize(store, converted)


@pytest.fixture(scope="session")
def ep_dir() -> Path:
    return realize(STORE_PATH.get(), energyplus_path())


@pytest.fixture(scope="session")
def weather_path() -> Path:
    zip_der = ASHRAE901_all_zip()
    filename = _pick_weather_file()
    return realize(STORE_PATH.get(), ExtractFromZip(zip_der, filename))


@pytest.mark.parametrize("building_type", BUILDING_TYPES)
def test_modified_building_simulates(
    building_type: BuildingType,
    ep_dir: Path,
    weather_path: Path,
) -> None:
    """An EnergyCodes building with parametric modifications must complete a
    design-day EnergyPlus simulation without fatal errors."""
    base_path = _convert_one(building_type)

    with open(base_path, "r") as f:
        epjson = json.load(f)

    apply_modifications(epjson, TEST_MODIFICATION)

    tmp = Path(tempfile.mkdtemp())
    modified_path = tmp / "modified.epjson"
    with open(modified_path, "w") as f:
        json.dump(epjson, f, indent=2)

    outdir = _run_eplus_designday(ep_dir, modified_path, weather_path)
    fatal = _has_fatal_errors(outdir)
    assert fatal is None, f"{building_type}: EnergyPlus fatal error: {fatal}"
