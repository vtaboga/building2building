import json
import logging
from pathlib import Path

from building2building.store import Derivation, OUTPUT, derivation

logger = logging.getLogger(__name__)


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
        for _var_key, var_data in epjson["Output:Variable"].items():
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
    """
    For dummy simulation.
    Ensure Output:EnergyManagementSystem is configured to generate .edd file.
    """
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
            obj["actuator_availability_dictionary_reporting"] = "Verbose"
            logger.info(
                f"Updated existing Output:EnergyManagementSystem '{key}' to Verbose reporting"
            )
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
            "ems_runtime_language_debug_output_level": "None",
        }
        logger.info("Added Output:EnergyManagementSystem for .edd file generation")

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


@derivation("with-tabular-output")
def AddTabularOutput(input: Path):
    """
    For dummy simulation.
    Ensure OutputControl:Files and OutputControl:Table:Style are configured to generate
    eplustbl.htm.
    """
    dst = OUTPUT.get()

    with open(input, "r") as f:
        epjson = json.load(f)

    # Ensure OutputControl:Files exists and has output_tabular enabled
    if "OutputControl:Files" not in epjson:
        epjson["OutputControl:Files"] = {}

    # Find or create OutputControl:Files entry
    files_key = None
    for key in epjson["OutputControl:Files"].keys():
        files_key = key
        break

    if files_key is None:
        files_key = "OutputControl:Files 1"
        epjson["OutputControl:Files"][files_key] = {}

    # Ensure output_tabular is set to "Yes"
    epjson["OutputControl:Files"][files_key]["output_tabular"] = "Yes"
    logger.info(
        f"Ensured OutputControl:Files '{files_key}' has output_tabular enabled"
    )

    # Ensure OutputControl:Table:Style exists and is set to HTML
    if "OutputControl:Table:Style" not in epjson:
        epjson["OutputControl:Table:Style"] = {}

    # Find or create OutputControl:Table:Style entry
    style_key = None
    for key in epjson["OutputControl:Table:Style"].keys():
        style_key = key
        break

    if style_key is None:
        style_key = "OutputControl:Table:Style 1"
        epjson["OutputControl:Table:Style"][style_key] = {}

    # Ensure column_separator is set to "HTML"
    epjson["OutputControl:Table:Style"][style_key]["column_separator"] = "HTML"
    logger.info(
        f"Ensured OutputControl:Table:Style '{style_key}' is set to HTML format"
    )

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


def add_hvac_meters(epjson_in: Derivation) -> Derivation:
    """Add HVAC energy meters to epJSON."""
    return AddHVACMeters(epjson_in)


def add_outdoor_air_meters(epjson_in: Derivation) -> Derivation:
    """Add outdoor air monitoring to epJSON."""
    return AddOutdoorAirMeters(epjson_in)


def add_edd_output(epjson_in: Derivation) -> Derivation:
    """Add EMS output to generate .edd file."""
    return AddEDDOutput(epjson_in)


def add_tabular_output(epjson_in: Derivation) -> Derivation:
    """Add tabular output configuration to generate eplustbl.htm file."""
    return AddTabularOutput(epjson_in)


def modify_timestep(epjson_in: Derivation, timesteps_per_hour: int = 4) -> Derivation:
    """Modify simulation timestep."""
    return ModifyTimestep(epjson_in, timesteps_per_hour)

