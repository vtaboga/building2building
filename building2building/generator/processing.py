import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeAlias,
    Union,
)

import geopandas as gpd
import pandas as pd
from minergym.ontology import Ontology

import building2building.env as env

logger = logging.getLogger(__name__)


@contextmanager
def timer(name="Code block"):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        logger.info(f"{name} took {end - start:.4f} seconds")


Transition: TypeAlias = Literal[
    "9.4.0-to-9.5.0",
    "9.5.0-to-9.6.0",
    "9.6.0-to-22.1.0",
    "22.1.0-to-22.2.0",
    "22.2.0-to-23.1.0",
    "23.1.0-to-23.2.0",
    "23.2.0-to-24.1.0",
]

transitions: list[Transition] = [
    "9.4.0-to-9.5.0",
    "9.5.0-to-9.6.0",
    "9.6.0-to-22.1.0",
    "22.1.0-to-22.2.0",
    "22.2.0-to-23.1.0",
    "23.1.0-to-23.2.0",
    "23.2.0-to-24.1.0",
]


def upgrade_idf(idf_in: Path, idf_out: Path, transition: Transition):
    from_version, to_version = transition.split("-to-")

    transition_dir = env.ENERGYPLUS_PATH.get() / "PreProcess" / "IDFVersionUpdater"

    transition_exe = (
        transition_dir
        / f"Transition-V{from_version.replace('.', '-')}-to-V{to_version.replace('.', '-')}"
    )
    if not transition_exe.exists():
        raise Exception(f"{transition_exe} does not exist")

    with tempfile.TemporaryDirectory() as temp:
        logger.debug(f"executing transition {transition} in {temp}")
        temp_path = Path(temp).resolve()

        temp_idf_in = temp_path / "in.idf"
        temp_idf_out = temp_path / "in.idfnew"

        from_idd_part = f"V{from_version.replace('.', '-')}-Energy+.idd"
        to_idd_part = f"V{to_version.replace('.', '-')}-Energy+.idd"

        from_idd = (transition_dir / Path(from_idd_part)).resolve()
        to_idd = (transition_dir / Path(to_idd_part)).resolve()

        if not from_idd.exists():
            raise Exception(f"{from_idd} doesn't exist")
        if not to_idd.exists():
            raise Exception(f"{to_idd} doesn't exist")

        (temp_path / from_idd_part).symlink_to(from_idd)
        (temp_path / to_idd_part).symlink_to(to_idd)

        shutil.copy(idf_in, temp_idf_in)

        # Run the transition executable from the temp_dir
        cmdline = [
            transition_exe,
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
            cwd=temp_path,
        )

        if result.stdout:
            logger.debug(f"Transition output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Transition errors: {result.stderr}")

        if not temp_idf_out.exists():
            raise Exception(f"{temp_idf_out} does not exist")

        shutil.copy(temp_idf_out, idf_out)


def specialized_upgrade_idf(t):
    x = partial(upgrade_idf, transition=t)
    x.__qualname__ = upgrade_idf.__qualname__ + "-" + t
    return x


def convert_idf(idf_path: Path, epjson_path: Path):
    # Path to the EnergyPlus executable
    converter = env.ENERGYPLUS_PATH.get() / "ConvertInputFormat"

    logger.info(f"Converting {idf_path} to epJSON format")

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        # Run the conversion

        temp_idf_path = temp_path / "in.idf"
        temp_epjson_path = temp_idf_path.with_suffix(".epJSON")

        shutil.copy(idf_path, temp_idf_path)

        result = subprocess.run(
            [converter, temp_idf_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp,
        )

        if result.stderr:
            logger.warning(f"Conversion warnings: {result.stderr}")
        if result.stdout:
            logger.warning(f"Conversion warnings: {result.stdout}")

        shutil.copy(temp_epjson_path, epjson_path)


def add_hvac_meters_to_epjson(epjson_path: Path, output_path: Path):
    """
    Check if HVAC energy consumption meters exist in an epJSON file.
    If not, add the meters and save the modified epJSON.

    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file (if None, will overwrite the input file)
    """

    # Load the epJSON file
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    # Check if we already have the necessary meters
    has_elec_hvac_meter = False
    has_gas_hvac_meter = False

    # First, check existing Output:Meter objects
    if "Output:Meter" in epjson:
        for meter_key, meter_data in epjson["Output:Meter"].items():
            if (
                meter_data.get("key_name") == "Electricity:HVAC"
                and meter_data.get("reporting_frequency") == "Timestep"
            ):
                has_elec_hvac_meter = True
                print("Found existing Electricity:HVAC meter with Timestep reporting.")

            if (
                meter_data.get("key_name") == "NaturalGas:HVAC"
                and meter_data.get("reporting_frequency") == "Timestep"
            ):
                has_gas_hvac_meter = True
                print("Found existing NaturalGas:HVAC meter with Timestep reporting.")

    # Check Output:Meter:MeterFileOnly objects as well
    if "Output:Meter:MeterFileOnly" in epjson:
        for meter_key, meter_data in epjson["Output:Meter:MeterFileOnly"].items():
            if (
                meter_data.get("key_name") == "Electricity:HVAC"
                and meter_data.get("reporting_frequency") == "Timestep"
            ):
                has_elec_hvac_meter = True
                print(
                    "Found existing Electricity:HVAC meter file only with Timestep reporting."
                )

            if (
                meter_data.get("key_name") == "NaturalGas:HVAC"
                and meter_data.get("reporting_frequency") == "Timestep"
            ):
                has_gas_hvac_meter = True
                print(
                    "Found existing NaturalGas:HVAC meter file only with Timestep reporting."
                )

    # Make sure the Output:Meter category exists
    if "Output:Meter" not in epjson:
        epjson["Output:Meter"] = {}

    # Add electricity HVAC meter if needed
    if not has_elec_hvac_meter:
        new_meter_name = f"Output:Meter:ElectricityHVAC"
        epjson["Output:Meter"][new_meter_name] = {
            "key_name": "Electricity:HVAC",
            "reporting_frequency": "Timestep",
        }
        print(f"Added Electricity:HVAC meter with Timestep reporting.")

    # Add natural gas HVAC meter if needed
    if not has_gas_hvac_meter:
        new_meter_name = f"Output:Meter:NaturalGasHVAC"
        epjson["Output:Meter"][new_meter_name] = {
            "key_name": "NaturalGas:HVAC",
            "reporting_frequency": "Timestep",
        }
        print(f"Added NaturalGas:HVAC meter with Timestep reporting.")

    with open(output_path, "w") as f:
        json.dump(epjson, f, indent=4)


def check_meter_availability(epjson_path: str) -> Dict[str, bool]:
    """
    Check the availability of all end-use meters in an epJSON file.
    Returns a dictionary of meter names and whether they exist.

    Args:
        epjson_path: Path to the epJSON file

    Returns:
        Dictionary mapping meter names to boolean (True if present)
    """
    # List of common end-use meters to check
    meter_list = [
        "Electricity:Facility",
        "Electricity:HVAC",
        "Electricity:Heating",
        "Electricity:Cooling",
        "Electricity:Fans",
        "Electricity:InteriorLights",
        "Electricity:ExteriorLights",
        "Electricity:InteriorEquipment",
        "NaturalGas:Facility",
        "NaturalGas:HVAC",
        "NaturalGas:Heating",
        "Fans:Electricity",
        "Cooling:Electricity",
        "Heating:Electricity",
        "Heating:NaturalGas",
    ]

    # Initialize results dictionary
    meter_availability = {meter: False for meter in meter_list}

    # Load the epJSON file
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    # Check Output:Meter objects
    if "Output:Meter" in epjson:
        for meter_key, meter_data in epjson["Output:Meter"].items():
            meter_name = meter_data.get("key_name")
            if meter_name in meter_availability:
                meter_availability[meter_name] = True

    # Check Output:Meter:MeterFileOnly objects
    if "Output:Meter:MeterFileOnly" in epjson:
        for meter_key, meter_data in epjson["Output:Meter:MeterFileOnly"].items():
            meter_name = meter_data.get("key_name")
            if meter_name in meter_availability:
                meter_availability[meter_name] = True

    return meter_availability


def add_outdoor_air_meters_to_epjson(epjson_path: Path, output_path: Path) -> None:
    """
    Check if outdoor air temperature and humidity output variables exist in an epJSON file.
    If not, add them and save the modified epJSON.

    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file
    """

    # Load the epJSON file
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    # Define outdoor air variables we want to check/add
    outdoor_vars = [
        # Variable Name, Key Value (usually "Environment")
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
                    print(
                        f"Found existing output variable: {var_name} with Timestep reporting."
                    )

    # Make sure the Output:Variable category exists
    if "Output:Variable" not in epjson:
        epjson["Output:Variable"] = {}

    # Add missing variables
    for var_name, key_value in outdoor_vars:
        if not has_outdoor_vars[var_name]:
            new_var_key = f"Output:Variable {var_name}"

            # Ensure the key is unique by adding a suffix if needed
            suffix = 1
            while new_var_key in epjson["Output:Variable"]:
                new_var_key = f"Output:Variable {var_name} {suffix}"
                suffix += 1

            epjson["Output:Variable"][new_var_key] = {
                "key_value": key_value,
                "variable_name": var_name,
                "reporting_frequency": "Timestep",
            }
            print(f"Added output variable: {var_name} with Timestep reporting.")

    with open(output_path, "w") as f:
        json.dump(epjson, f, indent=4)


def check_output_variables_availability(epjson_path: str) -> Dict[str, bool]:
    """
    Check the availability of common output variables in an epJSON file.
    Returns a dictionary of variable names and whether they exist.

    Args:
        epjson_path: Path to the epJSON file

    Returns:
        Dictionary mapping variable names to boolean (True if present)
    """
    # List of common variables to check
    variable_list = [
        "Site Outdoor Air Drybulb Temperature",
        "Site Outdoor Air Humidity Ratio",
        "Site Outdoor Air Relative Humidity",
        "Site Outdoor Air Wetbulb Temperature",
        "Site Outdoor Air Dewpoint Temperature",
        "Zone Air Temperature",
        "Zone Air Humidity Ratio",
        "Zone Air Relative Humidity",
        "Zone Thermostat Heating Setpoint Temperature",
        "Zone Thermostat Cooling Setpoint Temperature",
        "Zone Thermal Comfort Mean Radiant Temperature",
        "Zone People Occupant Count",
        "Zone Air System Sensible Heating Energy",
        "Zone Air System Sensible Cooling Energy",
        "System Node Temperature",
        "System Node Relative Humidity",
        "System Node Mass Flow Rate",
        "Fan Electricity Rate",
    ]

    # Initialize results dictionary
    var_availability = {var: False for var in variable_list}

    # Load the epJSON file
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    # Check Output:Variable objects
    if "Output:Variable" in epjson:
        for var_key, var_data in epjson["Output:Variable"].items():
            var_name = var_data.get("variable_name")
            if var_name in var_availability:
                var_availability[var_name] = True

    return var_availability


def add_outdoor_air_nodes_if_missing(epjson_path: Path, output_path: Path) -> None:
    """
    Check if outdoor air node exists and add it if missing.
    This ensures that outdoor air can be properly monitored.

    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = epjson_path

    # Load the epJSON file
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    # Check if we have an OutdoorAir:Node defined
    has_outdoor_air_node = (
        "OutdoorAir:Node" in epjson and len(epjson["OutdoorAir:Node"]) > 0
    )

    # Add OutdoorAir:Node if not present
    modified = False
    if not has_outdoor_air_node:
        # Create the OutdoorAir:Node category if it doesn't exist
        if "OutdoorAir:Node" not in epjson:
            epjson["OutdoorAir:Node"] = {}

        epjson["OutdoorAir:Node"]["Model Outdoor Air Node"] = {}
        print("Added OutdoorAir:Node for monitoring outdoor conditions.")
        modified = True
    else:
        print("Found existing OutdoorAir:Node.")

    # Check if we have an OutdoorAir:NodeList defined
    has_outdoor_air_nodelist = (
        "OutdoorAir:NodeList" in epjson and len(epjson["OutdoorAir:NodeList"]) > 0
    )

    # Add OutdoorAir:NodeList if not present
    if not has_outdoor_air_nodelist:
        # Create the OutdoorAir:NodeList category if it doesn't exist
        if "OutdoorAir:NodeList" not in epjson:
            epjson["OutdoorAir:NodeList"] = {}

        epjson["OutdoorAir:NodeList"]["OutdoorAir:NodeList"] = {
            "nodes": [{"node_or_nodelist_name": "Model Outdoor Air Node"}]
        }
        print("Added OutdoorAir:NodeList referencing outdoor air node.")
        modified = True
    else:
        print("Found existing OutdoorAir:NodeList.")

    # Save the modified epJSON if changes were made
    if modified:
        with open(output_path, "w") as f:
            json.dump(epjson, f, indent=4)
        print(f"Modified epJSON saved to {output_path}")
    else:
        print("No changes needed to outdoor air nodes configuration.")


def modify_timestep(
    epjson_path: str, output_path: Optional[str] = None, timesteps_per_hour: int = 4
) -> None:
    """
    Modifies the timestep in an epJSON file to a specific value.

    Args:
        epjson_path: Path to the input epJSON file
        output_path: Path to save the modified epJSON file (if None, will overwrite the input file)
        timesteps_per_hour: Number of timesteps per hour (default: 4, which is 15-minute timesteps)

    Returns:
        None
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = epjson_path

    # Load the epJSON file
    try:
        with open(epjson_path, "r") as f:
            epjson = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{epjson_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: '{epjson_path}' is not a valid JSON file.")
        return

    # Check for existing Timestep object
    timestep_found = False
    timestep_modified = False

    if "Timestep" in epjson:
        timestep_found = True
        # Process all timestep objects (usually there's just one)
        for timestep_key, timestep_data in epjson["Timestep"].items():
            current_timestep = timestep_data.get("number_of_timesteps_per_hour", 1)
            if current_timestep != timesteps_per_hour:
                print(
                    f"Changing timestep from {current_timestep} to {timesteps_per_hour} timesteps per hour"
                )
                epjson["Timestep"][timestep_key]["number_of_timesteps_per_hour"] = (
                    timesteps_per_hour
                )
                timestep_modified = True
            else:
                print(
                    f"Timestep already set to {timesteps_per_hour} timesteps per hour"
                )

    # If no Timestep object found, create one
    if not timestep_found:
        print(
            f"No Timestep object found. Creating one with {timesteps_per_hour} timesteps per hour"
        )
        epjson["Timestep"] = {
            "Timestep 1": {"number_of_timesteps_per_hour": timesteps_per_hour}
        }
        timestep_modified = True

    # Check if we need to update RunPeriod objects to match timestep
    # Some EnergyPlus simulations may have issues if RunPeriod doesn't align with timestep
    if "RunPeriod" in epjson:
        for runperiod_key, runperiod_data in epjson["RunPeriod"].items():
            # Just noting this - we don't need to change anything in RunPeriod for timestep
            pass

    # Save the modified epJSON if changes were made
    if timestep_modified:
        try:
            with open(output_path, "w") as f:
                json.dump(epjson, f, indent=4)
            print(f"Modified epJSON saved to {output_path}")
        except Exception as e:
            print(f"Error saving modified file: {e}")
    else:
        print("No changes were made to the epJSON file")


def add_setpoint_control_to_epjson(epjson_path: Path, output_path: Path):
    """
    Modifies an epJSON file to add controllable temperature setpoint schedules.

    Args:
        epjson_path (str): Path to the input epJSON file
        output_path (str, optional): Path to save the modified epJSON. If None, overwrites input file.
    """

    # Load the epJSON file
    with open(epjson_path, "r") as f:
        epjson = json.load(f)

    # Get thermostat setpoints used in the building
    thermostat_setpoints = get_temperature_setpoints(epjson_path)

    # Make sure Schedule:Compact exists in epjson
    if "Schedule:Compact" not in epjson:
        epjson["Schedule:Compact"] = {}

    def create_schedule_compact(temperature: float):
        """Helper function to create a schedule compact object in the correct format"""
        return {
            "data": [
                {"field": "Through: 12/31"},
                {"field": "For: AllDays"},
                {"field": "Until: 24:00"},
                {"field": temperature},
            ],
            "schedule_type_limits_name": "Temperature",
        }

    # Process each thermostat
    for control_type, setpoint_name in thermostat_setpoints:
        if control_type == "ThermostatSetpoint:DualSetpoint":
            # Get the original schedule names
            dual_setpoint = epjson["ThermostatSetpoint:DualSetpoint"][setpoint_name]

            # Create new schedule names
            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"

            # Update the thermostat to use new schedules
            dual_setpoint["cooling_setpoint_temperature_schedule_name"] = (
                cooling_schedule_name
            )
            dual_setpoint["heating_setpoint_temperature_schedule_name"] = (
                heating_schedule_name
            )

            # Create cooling setpoint schedule if it doesn't exist
            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = (
                    create_schedule_compact(25.0)
                )

            # Create heating setpoint schedule if it doesn't exist
            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = (
                    create_schedule_compact(20.0)
                )

        elif control_type == "ThermostatSetpoint:SingleHeating":
            # Create new schedule name
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"

            # Update the thermostat to use new schedule
            epjson["ThermostatSetpoint:SingleHeating"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = heating_schedule_name

            # Create heating setpoint schedule if it doesn't exist
            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = (
                    create_schedule_compact(20.0)
                )

        elif control_type == "ThermostatSetpoint:SingleCooling":
            # Create new schedule name
            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"

            # Update the thermostat to use new schedule
            epjson["ThermostatSetpoint:SingleCooling"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = cooling_schedule_name

            # Create cooling setpoint schedule if it doesn't exist
            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = (
                    create_schedule_compact(25.0)
                )

        elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
            # Create new schedule name for the single setpoint
            setpoint_schedule_name = f"{setpoint_name} Setpoint"

            # Update the thermostat to use new schedule
            epjson["ThermostatSetpoint:SingleHeatingOrCooling"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = setpoint_schedule_name

            # Create setpoint schedule if it doesn't exist
            if setpoint_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][setpoint_schedule_name] = (
                    create_schedule_compact(22.5)
                )

    # Save the modified epJSON
    with open(output_path, "w") as f:
        json.dump(epjson, f, indent=4)


def get_temperature_setpoints(epjson_path: Path) -> List[tuple]:
    """Analyzes an epJSON file to identify thermostat setpoints that are used to
    control zones.

    Args:
        epjson_path (str): Path to the epJSON file

    Returns:
        List[tuple]: List of tuples containing (thermostat_type, thermostat_name) for thermostats
                     that are used to control at least one zone

    """
    # Convert epJSON to RDF for querying
    ont = Ontology.from_json(epjson_path)

    # Query to find thermostats that are used in zone controls
    thermostat_query = """# -*- mode: sparql -*-
SELECT DISTINCT ?control_type ?setpoint_name
WHERE {
  # Find zone controls and their types
  ?control a "ZoneControl:Thermostat" .
  ?control idf:control_1_object_type ?control_type .
  ?control idf:control_1_name ?setpoint_name .

  # Make sure the control is used by at least one zone
  ?control idf:zone_or_zonelist_name ?zone_name .
}
"""

    # Execute query and process results
    results = []
    for row in ont.rdf.query(thermostat_query):
        control_type = str(row.control_type)
        setpoint_name = str(row.setpoint_name)
        results.append((control_type, setpoint_name))

    return results


def glue_surfaces(epjson_in: Path, epjson_out: Path):
    """Look at all the surfaces, and glue together those that have the same
    coordinates."""

    ont = Ontology.from_json(epjson_in)

    with open(epjson_in, "rb") as f:
        json_obj = json.load(f)

    surfaces = json_obj["BuildingSurface:Detailed"]

    mapping = ont.pointset_to_surfaceset()

    for surfaceset in mapping.values():
        if len(surfaceset) == 1:
            continue
        if len(surfaceset) != 2:
            raise Exception(f"wrong number of overlapping surfaces: {surfaceset}")

        surface_list = list(surfaceset)

        left_name = surface_list[0].toPython()
        right_name = surface_list[1].toPython()

        left = surfaces[left_name]
        right = surfaces[right_name]

        print(left_name, right_name)

        # Step 1: set outside_boundary_condition
        #         "outside_boundary_condition": "Surface",
        # "outside_boundary_condition_object": "ceiling_unit1_BackRow_BottomFloor",

        left["outside_boundary_condition"] = "Surface"
        right["outside_boundary_condition"] = "Surface"

        # Step 2: attach the correct surface

        left["outside_boundary_condition_object"] = right_name
        right["outside_boundary_condition_object"] = left_name

        with open(epjson_out, "w") as f:
            json.dump(json_obj, f, indent=4)
