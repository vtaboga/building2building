# Here all the steps necessary to transform a raw idf into a ready-to-go idf are
# expressed as BuildStep's so that they can be properly cached.
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame

from building2building.env import STORE_PATH
from building2building.store import (
    OUTPUT,
    ChildFile,
    Derivation,
    Realizable,
    Rename,
    derivation,
    expression,
    realize,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def chdir(path: Path):
    """
    Context manager to temporarily change the current working directory.
    Reimplementation of contextlib.chdir for Python < 3.11.
    """
    old_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


Transition: TypeAlias = Literal[
    "9.4.0-to-9.5.0",
    "9.5.0-to-9.6.0",
    "9.6.0-to-22.1.0",
    "22.1.0-to-22.2.0",
    "22.2.0-to-23.1.0",
    "23.1.0-to-23.2.0",
    "23.2.0-to-24.1.0",
]


@derivation("upgraded.idf")
def upgrade_idf(input: Path, transition_exe: Path, from_idd: Path, to_idd: Path):
    dst = OUTPUT.get()
    with tempfile.TemporaryDirectory() as tempdir:
        with chdir(Path(tempdir)):
            Path(from_idd.name).symlink_to(from_idd)
            Path(to_idd.name).symlink_to(to_idd)
            shutil.copy(input, "in.idf")

            temp_idf_out = Path(tempdir) / "in.idfnew"
            # Run the transition executable from the temp_dir
            cmdline = [
                str(transition_exe),
                "in.idf",
            ]

            logging.debug(f"Using cmd: {cmdline}")

            result = subprocess.run(
                cmdline,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"DISPLAY": ""},
                cwd=tempdir,
            )

        if result.stdout:
            logger.debug(f"Transition output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Transition errors: {result.stderr}")

        if not temp_idf_out.exists():
            raise Exception(
                f"failed to upgrade idf file from {from_idd.name} to {to_idd.name}"
            )

        shutil.copy(temp_idf_out, dst)


@derivation("building.epjson")
def ConvertIDF(input: Path, converter: Path):
    dst = OUTPUT.get()

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        temp_idf_path = temp_path / "in.idf"
        temp_epjson_path = temp_idf_path.with_suffix(".epJSON")

        shutil.copy(input, temp_idf_path)

        result = subprocess.run(
            [str(converter), str(temp_idf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp,
        )

        if result.stderr:
            print(result.stderr)
            # logger.info(f"Conversion warnings: {result.stderr}")
        if result.stdout:
            print(result.stdout)
            # logger.info(f"Conversion output: {result.stdout}")

        if not temp_epjson_path.exists():
            raise Exception("failed to convert idf to json")
        shutil.copy(temp_epjson_path, dst)


@derivation("with-meters")
def AddHVACMeters(input: Path):
    dst = OUTPUT.get()
    with open(input, "r") as f:
        epjson: dict = json.load(f)

    output_meter: dict = epjson.setdefault("Output:Meter", {})

    electricity_hvac = output_meter.setdefault("Output:Meter:ElectricityHVAC", {})
    electricity_hvac["key_name"] = "Electricity:HVAC"
    electricity_hvac["reporting_frequency"] = "Timestep"

    natural_gas_hvac = output_meter.setdefault("Output:Meter:NaturalGasHVAC", {})
    natural_gas_hvac["key_name"] = "NaturalGas:HVAC"
    natural_gas_hvac["reporting_frequency"] = "Timestep"

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


@derivation("outdoor-air")
def AddOutdoorAirMeters(input: Path):
    dst = OUTPUT.get()

    with open(input, "r") as f:
        epjson = json.load(f)

    # Define outdoor air variables we want to check/add
    outdoor_vars = [
        ("Site Outdoor Air Drybulb Temperature", "Environment"),
        ("Site Outdoor Air Humidity Ratio", "Environment"),
        ("Site Outdoor Air Relative Humidity", "Environment"),
        ("Site Outdoor Air Wetbulb Temperature", "Environment"),
        ("Site Outdoor Air Dewpoint Temperature", "Environment"),
    ]

    # Check if we already have the necessary output variables
    has_outdoor_vars = {var[0]: False for var in outdoor_vars}

    # Check existing Output:Variable objects
    if "Output:Variable" in epjson:
        for var_key, var_data in epjson["Output:Variable"].items():
            var_name = var_data.get("variable_name")
            for outdoor_var, _ in outdoor_vars:
                if (
                    var_name == outdoor_var
                    and var_data.get("reporting_frequency") == "Timestep"
                ):
                    has_outdoor_vars[outdoor_var] = True
                    logger.debug(f"Found existing output variable: {var_name}")

    # Make sure the Output:Variable category exists
    if "Output:Variable" not in epjson:
        epjson["Output:Variable"] = {}

    # Add missing variables
    for var_name, key_value in outdoor_vars:
        if not has_outdoor_vars[var_name]:
            new_var_key = f"Output:Variable {var_name}"

            # Ensure the key is unique
            suffix = 1
            while new_var_key in epjson["Output:Variable"]:
                new_var_key = f"Output:Variable {var_name} {suffix}"
                suffix += 1

            epjson["Output:Variable"][new_var_key] = {
                "key_value": key_value,
                "variable_name": var_name,
                "reporting_frequency": "Timestep",
            }
            logger.info(f"Added output variable: {var_name}")

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


@derivation("with-edd-output")
def AddEDDOutput(input: Path):
    """Ensure Output:EnergyManagementSystem is configured to generate .edd file."""
    dst = OUTPUT.get()

    with open(input, "r") as f:
        epjson = json.load(f)

    # Check if Output:EnergyManagementSystem already exists
    if "Output:EnergyManagementSystem" not in epjson:
        epjson["Output:EnergyManagementSystem"] = {}
    
    # Check if we already have an EDD output configured properly
    has_edd_output = False
    for key, obj in epjson["Output:EnergyManagementSystem"].items():
        current_val = obj.get("actuator_availability_dictionary_reporting", "None")
        if current_val != "Verbose":
            obj["actuator_availability_dictionary_reporting"] = "Verbose"  # Update to verbose to output meters
            logger.info(f"Updated existing Output:EnergyManagementSystem '{key}' to Verbose reporting")
        has_edd_output = True
        break 
    
    # Add it if not present 
    if not has_edd_output:
        edd_key = "Output:EnergyManagementSystem 1"
        suffix = 1
        while edd_key in epjson["Output:EnergyManagementSystem"]:
            suffix += 1
            edd_key = f"Output:EnergyManagementSystem {suffix}"
        
        epjson["Output:EnergyManagementSystem"][edd_key] = {
            "actuator_availability_dictionary_reporting": "Verbose",
            "internal_variable_availability_dictionary_reporting": "Verbose",
            "ems_runtime_language_debug_output_level": "None"
        }
        logger.info("Added Output:EnergyManagementSystem for .edd file generation")

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)

@derivation("timestep")
def ModifyTimestep(
    input: Path,
    timesteps_per_hour: int = 4,
):
    dst = OUTPUT.get()

    with open(input, "r") as f:
        epjson = json.load(f)

    # Check for existing Timestep object
    timestep_modified = False

    if "Timestep" in epjson:
        # Process all timestep objects (usually there's just one)
        for timestep_key, timestep_data in epjson["Timestep"].items():
            current_timestep = timestep_data.get("number_of_timesteps_per_hour", 1)
            if current_timestep != timesteps_per_hour:
                logger.info(
                    f"Changing timestep from {current_timestep} to {timesteps_per_hour}"
                )
                epjson["Timestep"][timestep_key]["number_of_timesteps_per_hour"] = (
                    timesteps_per_hour
                )
                timestep_modified = True
            else:
                logger.info(f"Timestep already set to {timesteps_per_hour}")
    else:
        # Create new Timestep object
        logger.info(
            f"Creating Timestep object with {timesteps_per_hour} timesteps per hour"
        )
        epjson["Timestep"] = {
            "Timestep 1": {"number_of_timesteps_per_hour": timesteps_per_hour}
        }

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


@derivation("setpoint-control")
def AddSetpointControl(
    input: Path,
):
    dst = OUTPUT.get()

    with open(input, "r") as f:
        epjson = json.load(f)

    # Get thermostat setpoints used in the building
    thermostat_setpoints = get_temperature_setpoints(epjson)

    # Make sure Schedule:Compact exists
    if "Schedule:Compact" not in epjson:
        epjson["Schedule:Compact"] = {}

    def create_schedule_compact(temperature: float):
        return {
            "data": [
                {"field": "Through: 12/31"},
                {"field": "For: AllDays"},
                {"field": "Until: 24:00"},
                {"field": temperature},
            ],
            "schedule_type_limits_name": "Temperature",
        }

    # Process each thermostat type
    for control_type, setpoint_name in thermostat_setpoints:
        if control_type == "ThermostatSetpoint:DualSetpoint":
            dual_setpoint = epjson["ThermostatSetpoint:DualSetpoint"][setpoint_name]

            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"

            dual_setpoint["cooling_setpoint_temperature_schedule_name"] = (
                cooling_schedule_name
            )
            dual_setpoint["heating_setpoint_temperature_schedule_name"] = (
                heating_schedule_name
            )

            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = (
                    create_schedule_compact(25.0)
                )

            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = (
                    create_schedule_compact(20.0)
                )

        elif control_type == "ThermostatSetpoint:SingleHeating":
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"
            epjson["ThermostatSetpoint:SingleHeating"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = heating_schedule_name

            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = (
                    create_schedule_compact(20.0)
                )

        elif control_type == "ThermostatSetpoint:SingleCooling":
            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"
            epjson["ThermostatSetpoint:SingleCooling"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = cooling_schedule_name

            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = (
                    create_schedule_compact(25.0)
                )

        elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
            setpoint_schedule_name = f"{setpoint_name} Setpoint"
            epjson["ThermostatSetpoint:SingleHeatingOrCooling"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = setpoint_schedule_name

            if setpoint_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][setpoint_schedule_name] = (
                    create_schedule_compact(22.5)
                )

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


def get_temperature_setpoints(epjson_data) -> list[tuple]:
    """Find thermostats that control zones."""
    results = []

    # Look for zone controls that reference thermostat setpoints
    if "ZoneControl:Thermostat" in epjson_data:
        for control_name, control_data in epjson_data["ZoneControl:Thermostat"].items():
            control_type = control_data.get("control_1_object_type")
            setpoint_name = control_data.get("control_1_name")

            if control_type and setpoint_name:
                results.append((control_type, setpoint_name))

    return results


@derivation("glued")
def GlueSurfaces(
    input: Path,
):
    dst = OUTPUT.get()
    with open(input, "r") as f:
        epjson = json.load(f)

    # This is a simplified version - would need minergym.ontology for full implementation
    surfaces = epjson.get("BuildingSurface:Detailed", {})

    # Basic surface matching by coordinates (simplified)
    # In the full version, this would use the ontology to find overlapping surfaces
    logger.info(f"Processing {len(surfaces)} surfaces for gluing")

    # For now, just copy the input to output
    # The full implementation would require the ontology library
    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


def convert_idf(idf_in: Derivation, energyplus_path: Realizable) -> Derivation:
    """Create an IDF to epJSON conversion build step."""
    converter = ChildFile(energyplus_path, "ConvertInputFormat")
    return ConvertIDF(idf_in, converter)


def add_hvac_meters(epjson_in: Derivation) -> Derivation:
    """Add HVAC energy meters to epJSON."""
    return AddHVACMeters(epjson_in)


def add_outdoor_air_meters(epjson_in: Derivation) -> Derivation:
    """Add outdoor air monitoring to epJSON."""
    return AddOutdoorAirMeters(epjson_in)

def add_edd_output(epjson_in: Derivation) -> Derivation:
    """Add EMS output to generate .edd file."""
    return AddEDDOutput(epjson_in)


def modify_timestep(epjson_in: Derivation, timesteps_per_hour: int = 4) -> Derivation:
    """Modify simulation timestep."""
    return ModifyTimestep(epjson_in, timesteps_per_hour)


def add_setpoint_control(epjson_in: Derivation) -> Derivation:
    """Add controllable temperature setpoints."""
    return AddSetpointControl(epjson_in)


def glue_surfaces(epjson_in: Derivation) -> Derivation:
    """Glue together overlapping surfaces."""
    return GlueSurfaces(epjson_in)


all_transitions: list[Transition] = [
    "9.4.0-to-9.5.0",
    "9.5.0-to-9.6.0",
    "9.6.0-to-22.1.0",
    "22.1.0-to-22.2.0",
    "22.2.0-to-23.1.0",
    "23.1.0-to-23.2.0",
    "23.2.0-to-24.1.0",
]


@expression()
def scan_upgraders(energyplus: Path) -> DataFrame:
    ivu = energyplus / "PreProcess/IDFVersionUpdater"

    pattern = re.compile(r"^Transition-V(\d+)-(\d+)-(\d+)-to-V(\d+)-(\d+)-(\d+)$")

    records = []

    for p in ivu.glob("Transition*"):
        if m := pattern.match(p.name):
            M1, m1, p1, M2, m2, p2 = m.groups()

            v1 = f"{M1}.{m1}.{p1}"
            v2 = f"{M2}.{m2}.{p2}"

            idd1 = ivu / f"V{M1}-{m1}-{p1}-Energy+.idd"
            idd2 = ivu / f"V{M2}-{m2}-{p2}-Energy+.idd"

            records.append((str(p), v1, v2, str(idd1), str(idd2)))

    return DataFrame(
        records,
        columns=["path", "src_version", "dst_version", "src_idd", "dst_idd"],
    )


def upgrade(
    input_file: Derivation, energyplus_path: Realizable, src_version: str
) -> Derivation:
    # Multi-step upgrade process

    upgraders: DataFrame = realize(STORE_PATH.get(), scan_upgraders(energyplus_path))

    # Chain upgrades
    current = input_file
    current_version = src_version

    # Ensure the given src_version is even a valid version at all
    valid_versions = set(upgraders.src_version) | set(upgraders.dst_version)

    if src_version not in valid_versions:
        raise Exception(
            f"src_version ({src_version}) is not valid. valid versions: {valid_versions}"
        )

    while True:
        possible_upgraders = upgraders[upgraders.src_version == current_version]
        if len(possible_upgraders) == 0:
            break
        upgrader = possible_upgraders.iloc[0]

        current = upgrade_idf(
            current, Path(upgrader.path), Path(upgrader.src_idd), Path(upgrader.dst_idd)
        )

        current_version = upgrader.dst_version

    return current


def create_complete_pipeline(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
) -> Derivation:
    """Create a complete processing pipeline from raw IDF to ready-to-go epJSON."""

    current = upgrade(input_file, energyplus_path, src_version)

    # Convert to epJSON
    current = convert_idf(current, energyplus_path)

    # Add meters and monitoring
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)
    current = add_edd_output(current)

    # Configure simulation
    current = modify_timestep(current, timesteps_per_hour=4)
    current = add_setpoint_control(current)

    # Make the name useful
    current = Rename("building.epjson", current)

    # # Final processing
    # current = glue_surfaces(current)

    return current


@derivation("linked.epjson")
def link_in_schedule(epjson_file: Path, csv_file: Path):
    dst = OUTPUT.get()
    with open(epjson_file, "r") as f:
        epjson = json.load(f)

    for k, v in epjson["Schedule:File"].items():
        v["file_name"] = str(csv_file)

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


@derivation("simulation-outputs")
def run_simulation(ep_path: Path, epjson: Path, eps: Path):
    """Run an EnergyPlus simulation and save the output files."""
    out = OUTPUT.get()

    tmp = Path(tempfile.mkdtemp())
    cmd = [
        str(ep_path / "energyplus"),
        "-d",
        str(tmp),
        "-w",
        str(eps),
        "-x",
        str(epjson),
    ]

    subprocess.run(cmd, check=True)

    htm_file = tmp / "eplustbl.htm"
    edd_file = tmp / "eplusout.edd"

    if not htm_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplustbl.htm")
    if not edd_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplusout.edd")
    
    shutil.copy(htm_file, out / "eplustbl.htm")
    shutil.copy(edd_file, out / "eplusout.edd")


def eplustbl(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplustbl.htm file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplustbl.htm")


def eddfile(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplusout.edd file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplusout.edd")


def get_net_conditioned_area(html_path: Path) -> float:
    """
    Extract 'Net Conditioned Building Area' [m²] from an EnergyPlus
    HTML summary report (eplusout.html / eplusbl.htm / eplustbl.htm).

    Returns:
        area_m2 (float) or None if not found.
    """
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Find <b>Building Area</b> and get its following <table>
    table_tag = None
    for b in soup.find_all("b"):
        if "Building Area" in b.text:
            table_tag = b.find_next("table")
            break
    if table_tag is None:
        raise Exception("could not find conditionned area")

    # Use StringIO to satisfy pandas future behavior
    df = pd.read_html(StringIO(str(table_tag)))[0]

    # Clean headers robustly (convert to strings first)
    df.columns = [str(col).strip() for col in df.columns]

    # Look for the "Net Conditioned Building Area" row (case-insensitive)
    mask = (
        df.iloc[:, 0]
        .astype(str)
        .str.contains("Net Conditioned Building Area", case=False)
    )
    if not mask.any():
        raise Exception("could not find conditionned area")

    # Extract and return the numeric value (2nd column)
    value = float(df.loc[mask].iloc[0, 1])
    return value


def get_warmup_days(html_path: Path) -> float:
    """Extract the number of warm-up days from an EnergyPlus HTML summary file."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    table_tag = None
    for b in soup.find_all("b"):
        if "Environment:WarmupDays" in b.text:
            table_tag = b.find_next("table")
            break
    if table_tag is None:
        raise Exception("could not read warmup days")

    # Read without trusting header detection; promote first row to header if needed
    df = pd.read_html(StringIO(str(table_tag)), header=None)[0]
    # If columns look generic (e.g., '0', '1'), use first row as header
    generic_cols = all(str(c).isdigit() for c in df.columns)
    if generic_cols and len(df) > 0:
        new_cols = [str(c).strip() for c in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = new_cols
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Normalize and find the warmup column
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    colmap = {norm(c): c for c in df.columns}
    target = None
    for key in ("numberofwarmupdays", "warmupdays"):
        if key in colmap:
            target = colmap[key]
            break
    if target is None:
        # Fallback: any column mentioning both warmup and days
        for c in df.columns:
            nc = norm(c)
            if "warmup" in nc and "day" in nc:
                target = c
                break
    if target is None:
        raise Exception("could not read warmup days")

    # Take the first numeric value in that column
    def to_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return None

    series = df[target].map(to_float).dropna()

    if series.empty:
        raise Exception("could not read warmup days")
    warmup_days = float(series.iloc[0])
    return warmup_days


def get_hvac_actuators(edd_path: Path) -> list[dict[str, str]]:
    """Extract HVAC-related actuator names from an EnergyPlus .edd file.
    
    Searches for actuators related to:
    - Coil speed control (heating/cooling coils)
    - Fan air mass flow rate control
    - UnitarySystem air flow rate controls
    - AirTerminal mass flow rate controls
    
    Returns:
        List of dictionaries containing actuator information for get_actuator_handle().
        Each dictionary has keys: 'component_name', 'component_type', 'control_type', 'units'
    """
    
    hvac_actuators = []
    
    # Read the .edd file
    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    # Keywords to identify HVAC actuators (case-insensitive search)
    hvac_keywords = [
        # Coil and speed controls
        "Coil Speed Control",
        # Fan controls
        "Fan,Fan Air Mass Flow Rate",
        "Fan Air Mass Flow Rate",
        # UnitarySystem air flow controls
        "UnitarySystem,Autosized Supply Air Flow Rate",
        # AirTerminal controls
        "AirTerminal",
        # Exclude schedules and other non-direct controls
    ]
    
    # Lines to exclude (schedules, not direct HVAC equipment controls)
    exclude_keywords = [
        "Schedule:Year",
        "Schedule:File",
        "Schedule:Compact",
        "Schedule:Constant",
        "ElectricEquipment",
        "OtherEquipment",
        "Surface,",
        "Weather Data",
        "Material,",
        "People,",
        "Lights,",
        "Zone,",
        "System Node Setpoint",
        "Plant Component",
        "Autosized",
    ]
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip comments and empty lines
        if not line_stripped or line_stripped.startswith("!"):
            continue
        
        # Check if line contains HVAC-related keywords
        line_lower = line_stripped.lower()
        
        # First check if line should be excluded
        should_exclude = any(excl.lower() in line_lower for excl in exclude_keywords)
        if should_exclude:
            continue
        
        # Check if line contains any HVAC keywords
        is_hvac = any(keyword.lower() in line_lower for keyword in hvac_keywords)
        
        # Also check for specific component types that are HVAC-related
        if not is_hvac:
            # Additional patterns for HVAC equipment
            if "unitarysystem," in line_lower:
                is_hvac = True
            elif "airterminal:" in line_lower:
                is_hvac = True
            elif ("fan," in line_lower and "mass flow" in line_lower):
                is_hvac = True
            elif "coil" in line_lower and ("speed" in line_lower or "stage" in line_lower):
                is_hvac = True
        
        if is_hvac:
            # Parse the actuator line
            # Format: EnergyManagementSystem:Actuator Available,<Component Name>,<Component Type>,<Control Type>,<Units>
            parts = line_stripped.split(",", maxsplit=4)
            
            if len(parts) >= 5:
                actuator_dict = {
                    "component_name": parts[1].strip(),
                    "component_type": parts[2].strip(),
                    "control_type": parts[3].strip(),
                    "units": parts[4].strip(),
                }
                hvac_actuators.append(actuator_dict)
    
    return hvac_actuators