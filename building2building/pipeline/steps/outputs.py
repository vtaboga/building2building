import json
import logging
from pathlib import Path

from building2building.store import OUTPUT, Derivation, Realizable, derivation

logger = logging.getLogger(__name__)


@derivation("with-meters.epjson")
def add_hvac_meters(input: Path):
    """Add HVAC energy meters to epJSON."""
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


@derivation("outdoor-air.epjson")
def add_outdoor_air_meters(input: Path):
    """Add outdoor air monitoring to epJSON."""
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


@derivation("with-edd-output.epjson")
def add_edd_output(input: Path):
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


@derivation("with-tabular-output.epjson")
def add_tabular_output(input: Path):
    """Ensure OutputControl:Files and OutputControl:Table:Style are configured to generate eplustbl.htm."""
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
    logger.info(f"Ensured OutputControl:Files '{files_key}' has output_tabular enabled")

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


@derivation("with-sqlite-output.epjson")
def add_sqlite_output(input: Path):
    """Ensure Output:SQLite is configured to generate eplusout.sql file."""
    dst = OUTPUT.get()

    with open(input, "r") as f:
        epjson = json.load(f)

    # Ensure Output:SQLite exists
    if "Output:SQLite" not in epjson:
        epjson["Output:SQLite"] = {}

    # Find or create Output:SQLite entry
    sqlite_key = None
    for key in epjson["Output:SQLite"].keys():
        sqlite_key = key
        break

    if sqlite_key is None:
        sqlite_key = "Output:SQLite 1"
        epjson["Output:SQLite"][sqlite_key] = {}

    # Set option_type to "SimpleAndTabular" (most comprehensive)
    # Options: "Simple" | "SimpleAndTabular"
    epjson["Output:SQLite"][sqlite_key]["option_type"] = "SimpleAndTabular"
    logger.info(f"Ensured Output:SQLite '{sqlite_key}' is configured")

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


@derivation("timestep.epjson")
def modify_timestep(
    input: Path,
    timesteps_per_hour: int = 12,
):
    """Modify simulation timestep."""
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
                epjson["Timestep"][timestep_key][
                    "number_of_timesteps_per_hour"
                ] = timesteps_per_hour
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


@derivation("run-period.epjson")
def modify_run_period(
    input: Path,
    begin_day_of_month: int,
    begin_month: int,
    end_day_of_month: int,
    end_month: int,
):
    """Modify run period."""
    dst = OUTPUT.get()

    with open(input, "r") as f:
        epjson = json.load(f)

    template_run_period = {
        "Run Period 1": {
            "apply_weekend_holiday_rule": "No",
            "begin_day_of_month": begin_day_of_month,
            "begin_month": begin_month,
            "begin_year": 2023,
            "day_of_week_for_start_day": "Sunday",
            "end_day_of_month": end_day_of_month,
            "end_month": end_month,
            "end_year": 2023,
            "use_weather_file_daylight_saving_period": "No",
            "use_weather_file_holidays_and_special_days": "No",
            "use_weather_file_rain_indicators": "Yes",
            "use_weather_file_snow_indicators": "Yes",
        }
    }

    run_period_obj = epjson.get("RunPeriod")
    if run_period_obj is None:
        run_period_obj = template_run_period
        epjson["RunPeriod"] = run_period_obj
    if not isinstance(run_period_obj, dict):
        raise TypeError(
            f"Expected epJSON['RunPeriod'] to be a dict, got {type(run_period_obj)}"
        )

    # Rules:
    # - If "Run Period 1" exists, do not replace the object; only update its dates.
    # - If there is exactly one run period (any key), rename it to "Run Period 1".
    # - If there are multiple run periods, keep only the first one, rename it to
    #   "Run Period 1", and delete the others.
    # - If empty/missing, use the template.

    if not run_period_obj:
        run_period_obj.update(template_run_period)

    if "Run Period 1" in run_period_obj:
        rp1 = run_period_obj["Run Period 1"]
        if not isinstance(rp1, dict):
            raise TypeError(
                f"Expected epJSON['RunPeriod']['Run Period 1'] to be a dict, got {type(rp1)}"
            )
    else:
        # Choose the first existing run period (in file/insertion order).
        first_key, rp1 = next(iter(run_period_obj.items()))
        if not isinstance(rp1, dict):
            raise TypeError(
                f"Expected epJSON['RunPeriod'][{first_key!r}] to be a dict, got {type(rp1)}"
            )
        run_period_obj.clear()
        run_period_obj["Run Period 1"] = rp1

    rp1["begin_day_of_month"] = begin_day_of_month
    rp1["begin_month"] = begin_month
    rp1["end_day_of_month"] = end_day_of_month
    rp1["end_month"] = end_month

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


def add_all_outputs(epjson_in: Realizable) -> Derivation:
    """Add all output configurations needed for dummy simulations."""
    out = epjson_in
    out = add_hvac_meters(out)
    out = add_outdoor_air_meters(out)
    out = add_edd_output(out)
    out = add_tabular_output(out)
    out = add_sqlite_output(out)
    return out
