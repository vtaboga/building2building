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
from typing import Any, Literal, TypeAlias, Iterable, Iterator
from typing import Sequence

import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame
import sqlite3

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


@derivation("with-tabular-output")
def AddTabularOutput(input: Path):
    """
    For dummy simulation.
    Ensure OutputControl:Files and OutputControl:Table:Style are configured to generate eplustbl.htm.
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
    logger.info(f"Ensured OutputControl:Table:Style '{style_key}' is set to HTML format")

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


def _gather_unitary_and_coil_outlet_nodes(epjson: dict[str, Any]) -> list[str]:
    """
    Collect node names that are useful for verifying setpoint-based unitary control.

    We target:
    - `AirLoopHVAC:UnitarySystem.air_outlet_node_name`
    - The *outlet* node of referenced cooling/heating/supplemental coils (when present).
    """

    def _read_str(obj: Any, key: str) -> str | None:
        if not isinstance(obj, dict):
            return None
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    def _lookup_object(epjson0: dict[str, Any], obj_type: str, obj_name: str) -> dict[str, Any] | None:
        table = epjson0.get(obj_type)
        if not isinstance(table, dict):
            return None
        obj = table.get(obj_name)
        if not isinstance(obj, dict):
            return None
        return obj

    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return []

    nodes: list[str] = []

    for _sys_name, sys_any in unitary_any.items():
        if not isinstance(sys_any, dict):
            continue

        # System outlet node (supply air leaving the unitary system).
        if n := _read_str(sys_any, "air_outlet_node_name"):
            nodes.append(n)

        # Coil outlet nodes, if accessible from referenced objects.
        for prefix in ("cooling", "heating", "supplemental_heating"):
            obj_type = _read_str(sys_any, f"{prefix}_coil_object_type")
            obj_name = _read_str(sys_any, f"{prefix}_coil_name")
            if not obj_type or not obj_name:
                continue

            obj = _lookup_object(epjson, obj_type, obj_name)
            if obj is None:
                continue

            # Different coil objects use slightly different naming conventions.
            for outlet_key in ("air_outlet_node_name", "outlet_node_name"):
                if n := _read_str(obj, outlet_key):
                    nodes.append(n)
                    break

    # Stable ordering + dedup
    out: list[str] = []
    seen: set[str] = set()
    for n in nodes:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _gather_unitary_supply_fans(epjson: dict[str, Any]) -> list[str]:
    """Collect `AirLoopHVAC:UnitarySystem.supply_fan_name` values (deduped)."""
    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return []

    fans: list[str] = []
    for _sys_name, sys_any in unitary_any.items():
        if not isinstance(sys_any, dict):
            continue
        fan = sys_any.get("supply_fan_name")
        if isinstance(fan, str) and fan.strip():
            fans.append(fan.strip())

    out: list[str] = []
    seen: set[str] = set()
    for f in fans:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def get_unitary_air_outlet_node_names(epjson: dict[str, Any]) -> list[str]:
    """Return `AirLoopHVAC:UnitarySystem.air_outlet_node_name` values (deduped)."""
    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return []

    nodes: list[str] = []
    for _sys_name, sys_any in unitary_any.items():
        if not isinstance(sys_any, dict):
            continue
        n = sys_any.get("air_outlet_node_name")
        if isinstance(n, str) and n.strip():
            nodes.append(n.strip())

    out: list[str] = []
    seen: set[str] = set()
    for n in nodes:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def add_node_setpoint_diagnostics_inplace(
    epjson: dict[str, Any],
    *,
    reporting_frequency: str = "Timestep",
) -> None:
    """
    Add `Output:Variable` requests that let you verify:
    - what **node temperature setpoints** are being used
    - what the **actual node temperatures** are

    After running EnergyPlus, these appear in `eplusout.csv` / `eplusout.sql` as
    time series columns.
    """
    nodes = _gather_unitary_and_coil_outlet_nodes(epjson)
    fans = _gather_unitary_supply_fans(epjson)
    if not nodes and not fans:
        return

    outvars = epjson.setdefault("Output:Variable", {})
    if not isinstance(outvars, dict):
        raise TypeError("epjson['Output:Variable'] must be a dict if present")

    def _has(var_name: str, key_value: str) -> bool:
        for _k, obj_any in outvars.items():
            if not isinstance(obj_any, dict):
                continue
            if obj_any.get("variable_name") != var_name:
                continue
            if obj_any.get("key_value") != key_value:
                continue
            if obj_any.get("reporting_frequency") != reporting_frequency:
                continue
            return True
        return False

    desired_node_vars = (
        "System Node Setpoint Temperature",
        "System Node Temperature",
        "System Node Mass Flow Rate",
    )

    desired_fan_vars = ("Fan Air Mass Flow Rate",)

    for node in nodes:
        for var_name in desired_node_vars:
            if _has(var_name, node):
                continue
            base = f"Output:Variable {var_name} {node}"
            key = _ensure_unique_object_name(outvars, base)
            outvars[key] = {
                "key_value": node,
                "variable_name": var_name,
                "reporting_frequency": reporting_frequency,
            }

    for fan in fans:
        for var_name in desired_fan_vars:
            if _has(var_name, fan):
                continue
            base = f"Output:Variable {var_name} {fan}"
            key = _ensure_unique_object_name(outvars, base)
            outvars[key] = {
                "key_value": fan,
                "variable_name": var_name,
                "reporting_frequency": reporting_frequency,
            }


def get_b2b_scheduled_setpoint_schedule_names(epjson: dict[str, Any]) -> list[str]:
    """
    Return the `schedule_name` values used by our `B2B Node Temp SPM ...` setpoint managers.

    These schedule names correspond to `Schedule:Constant` objects that can be controlled
    via EMS `Schedule Value` actuators found in `.edd`.
    """
    spm = epjson.get("SetpointManager:Scheduled", {})
    if not isinstance(spm, dict):
        return []
    out: list[str] = []
    for k, obj_any in spm.items():
        if not isinstance(k, str) or not k.startswith("B2B Node Temp SPM "):
            continue
        if not isinstance(obj_any, dict):
            continue
        sched = obj_any.get("schedule_name")
        if isinstance(sched, str) and sched.strip():
            out.append(sched.strip())
    # stable dedup
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


@derivation("node-setpoint-diagnostics")
def AddNodeSetpointDiagnostics(input: Path) -> None:
    dst = OUTPUT.get()
    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    add_node_setpoint_diagnostics_inplace(epjson, reporting_frequency="Timestep")

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)


def add_node_setpoint_diagnostics(epjson_in: Derivation) -> Derivation:
    """Add `Output:Variable` requests for node setpoint temperature diagnostics."""
    return AddNodeSetpointDiagnostics(epjson_in)


def _ensure_unique_object_name(existing: dict[str, Any], base: str) -> str:
    """
    Generate a unique key for an epJSON object map (e.g. Schedule:Constant),
    given a desired base name.
    """
    name = base
    if name not in existing:
        return name
    suffix = 2
    while f"{base} {suffix}" in existing:
        suffix += 1
    return f"{base} {suffix}"


def _resolve_onoff_schedule_type_limits_name(epjson: dict[str, Any]) -> str:
    """
    Return a ScheduleTypeLimits name suitable for a 0/1 availability schedule.
    Creates one if none exist.
    """
    stl = epjson.setdefault("ScheduleTypeLimits", {})
    if not isinstance(stl, dict):
        raise TypeError("epjson['ScheduleTypeLimits'] must be a dict if present")

    for preferred in ("OnOff", "OnOff 1", "On-Off", "On/Off"):
        if preferred in stl:
            return preferred

    for k in stl.keys():
        if "onoff" in str(k).replace(" ", "").lower():
            return str(k)

    if stl:
        return str(next(iter(stl.keys())))

    # Create a minimal discrete availability type limit.
    name = "B2B OnOff"
    stl[name] = {
        "lower_limit_value": 0,
        "upper_limit_value": 1,
        "numeric_type": "Discrete",
        "unit_type": "Availability",
    }
    return name


def _resolve_temperature_schedule_type_limits_name(epjson: dict[str, Any]) -> str:
    """
    Return a ScheduleTypeLimits name suitable for a temperature schedule.
    Creates one if none exist.
    """
    stl = epjson.setdefault("ScheduleTypeLimits", {})
    if not isinstance(stl, dict):
        raise TypeError("epjson['ScheduleTypeLimits'] must be a dict if present")

    for preferred in ("Temperature", "Temperature 1"):
        if preferred in stl:
            return preferred

    for k in stl.keys():
        if "temperature" in str(k).replace(" ", "").lower():
            return str(k)

    # Create a minimal temperature type limit.
    name = "B2B Temperature"
    stl[name] = {
        "lower_limit_value": -100,
        "upper_limit_value": 200,
        "numeric_type": "Continuous",
        "unit_type": "Temperature",
    }
    return name


@derivation("baseboard-availability-control")
def AddBaseboardAvailabilityControl(input: Path) -> Path:
    """
    If the model contains ZoneHVAC:Baseboard:Convective:Electric, make each baseboard's
    Availability Schedule uniquely controllable via EMS by assigning it a dedicated
    Schedule:Constant (default=1).
    """
    dst = OUTPUT.get()

    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    baseboards = epjson.get("ZoneHVAC:Baseboard:Convective:Electric", {})
    if not isinstance(baseboards, dict) or not baseboards:
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(epjson, f, indent=4)
        return dst

    stl_name = _resolve_onoff_schedule_type_limits_name(epjson)
    sched_const = epjson.setdefault("Schedule:Constant", {})
    if not isinstance(sched_const, dict):
        raise TypeError("epjson['Schedule:Constant'] must be a dict if present")

    for bb_name, bb_obj_any in baseboards.items():
        if not isinstance(bb_obj_any, dict):
            continue

        base_sched_name = f"B2B Baseboard Availability {bb_name}"
        sched_name = _ensure_unique_object_name(sched_const, base_sched_name)

        if sched_name not in sched_const:
            sched_const[sched_name] = {
                "hourly_value": 1,
                "schedule_type_limits_name": stl_name,
            }

        bb_obj_any["availability_schedule_name"] = sched_name

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)
    return dst


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
                    create_schedule_compact(40.0)  # Default to extreme value to allow for low level control
                )

            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = (
                    create_schedule_compact(10.0)  # Default to extreme value to allow for low level control
                )

        elif control_type == "ThermostatSetpoint:SingleHeating":
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"
            epjson["ThermostatSetpoint:SingleHeating"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = heating_schedule_name

            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = (
                    create_schedule_compact(10.0)  # Default to extreme value to allow for low level control
                )

        elif control_type == "ThermostatSetpoint:SingleCooling":
            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"
            epjson["ThermostatSetpoint:SingleCooling"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = cooling_schedule_name

            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = (
                    create_schedule_compact(40.0)  # Default to extreme value to allow for low level control
                )

        elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
            
            raise NotImplementedError(
                "ThermostatSetpoint:SingleHeatingOrCooling not supported yet"
            )
        
            # remove supported for now to avoid problems with low level control
        
            # setpoint_schedule_name = f"{setpoint_name} Setpoint"
            # epjson["ThermostatSetpoint:SingleHeatingOrCooling"][setpoint_name][
            #     "setpoint_temperature_schedule_name"
            # ] = setpoint_schedule_name

            # if setpoint_schedule_name not in epjson["Schedule:Compact"]:
            #     epjson["Schedule:Compact"][setpoint_schedule_name] = (
            #         create_schedule_compact(22.5)
            #     )

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


def add_tabular_output(epjson_in: Derivation) -> Derivation:
    """Add tabular output configuration to generate eplustbl.htm file."""
    return AddTabularOutput(epjson_in)


def modify_timestep(epjson_in: Derivation, timesteps_per_hour: int = 4) -> Derivation:
    """Modify simulation timestep."""
    return ModifyTimestep(epjson_in, timesteps_per_hour)


def add_setpoint_control(epjson_in: Derivation) -> Derivation:
    """Add controllable temperature setpoints."""
    return AddSetpointControl(epjson_in)


def glue_surfaces(epjson_in: Derivation) -> Derivation:
    """Glue together overlapping surfaces."""
    return GlueSurfaces(epjson_in)


def set_unitary_systems_to_setpoint_control_inplace(epjson: dict[str, Any]) -> None:
    """
    In-place edit: set all `AirLoopHVAC:UnitarySystem` objects to SetPoint control.

    Rationale:
    - Many residential mini-split models use `AirLoopHVAC:UnitarySystem.control_type = Load`
      with a controlling zone thermostat. In that mode, changing coil node setpoints and fan
      flow rate often has little effect if the thermostat is not calling for conditioning.
    - For our control mode (fan airflow + coil/node temperature setpoints), we want the unitary
      system to track a supply air temperature setpoint instead of thermostat load.
    """
    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return

    # Ensure we have a valid fan operating mode schedule for unitary systems.
    # EnergyPlus expects values in (0, 1] for AirLoopHVAC:UnitarySystem
    # Supply Air Fan Operating Mode Schedule Name (0 can trigger severe errors).
    stl_name = _resolve_onoff_schedule_type_limits_name(epjson)
    sched_const = epjson.setdefault("Schedule:Constant", {})
    if not isinstance(sched_const, dict):
        raise TypeError("epjson['Schedule:Constant'] must be a dict if present")

    fan_mode_schedule_base = "B2B Unitary Fan Operating Mode"
    fan_mode_schedule = _ensure_unique_object_name(sched_const, fan_mode_schedule_base)
    if fan_mode_schedule not in sched_const:
        # Use 1.0 (continuous fan) to satisfy strict validation.
        sched_const[fan_mode_schedule] = {
            "hourly_value": 1,
            "schedule_type_limits_name": stl_name,
        }

    for _name, obj_any in unitary_any.items():
        if not isinstance(obj_any, dict):
            continue
        obj_any["control_type"] = "SetPoint"
        obj_any["supply_air_fan_operating_mode_schedule_name"] = fan_mode_schedule


@derivation("unitary-setpoint-control")
def SetUnitarySystemsToSetpointControl(input: Path) -> None:
    dst = OUTPUT.get()
    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    set_unitary_systems_to_setpoint_control_inplace(epjson)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)


def set_unitary_systems_to_setpoint_control(epjson_in: Derivation) -> Derivation:
    return SetUnitarySystemsToSetpointControl(epjson_in)


def _node_setpoint_category(node_name: str) -> str:
    up = str(node_name).strip().upper()
    if "COOLING COIL NODE" in up:
        return "cooling_coil"
    if "HEATING COIL" in up and "SUPPLEMENTAL" not in up:
        return "heating_coil"
    if "SUPPLEMENTAL COIL" in up:
        return "supplemental_coil"
    return "unitary_outlet"


def ensure_scheduled_node_temperature_setpoints_inplace(
    epjson: dict[str, Any],
    *,
    temperature_c: float = 22.0,
) -> None:
    """
    Ensure the nodes we care about are controlled by `SetpointManager:Scheduled` objects
    whose `schedule_name` is a `Schedule:Constant`.

    Why this is more robust than directly actuating `System Node Setpoint`:
    - Many models already have SetpointManagers on these nodes. If we directly actuate
      the node setpoint, a SetpointManager may overwrite our value later in the timestep.
    - If we instead actuate the *schedule value* used by the SetpointManager, we become
      the authoritative source of the node setpoint EnergyPlus uses and reports.

    This step covers:
    - Unitary system outlet nodes (`AirLoopHVAC:UnitarySystem.air_outlet_node_name`)
    - Coil outlet nodes referenced by the unitary system (when discoverable from epJSON)
    """
    nodes = _gather_unitary_and_coil_outlet_nodes(epjson)
    if not nodes:
        return

    stl_name = _resolve_temperature_schedule_type_limits_name(epjson)
    sched_const = epjson.setdefault("Schedule:Constant", {})
    if not isinstance(sched_const, dict):
        raise TypeError("epjson['Schedule:Constant'] must be a dict if present")

    spm = epjson.setdefault("SetpointManager:Scheduled", {})
    if not isinstance(spm, dict):
        raise TypeError("epjson['SetpointManager:Scheduled'] must be a dict if present")

    # Track nodes already covered by any scheduled setpoint manager.
    covered_nodes: set[str] = set()
    for _k, obj_any in spm.items():
        if not isinstance(obj_any, dict):
            continue
        node = obj_any.get("setpoint_node_or_nodelist_name")
        if isinstance(node, str) and node.strip():
            covered_nodes.add(node.strip())

    for node in nodes:
        if node in covered_nodes:
            continue

        cat = _node_setpoint_category(node)
        schedule_base = f"B2B Node Temp SP {cat} {node}"
        schedule_name = _ensure_unique_object_name(sched_const, schedule_base)
        if schedule_name not in sched_const:
            sched_const[schedule_name] = {
                "hourly_value": float(temperature_c),
                "schedule_type_limits_name": stl_name,
            }

        spm_base = f"B2B Node Temp SPM {cat} {node}"
        spm_name = _ensure_unique_object_name(spm, spm_base)
        spm[spm_name] = {
            "control_variable": "Temperature",
            "schedule_name": schedule_name,
            "setpoint_node_or_nodelist_name": node,
        }
        covered_nodes.add(node)


@derivation("scheduled-node-temp-setpoints")
def EnsureScheduledNodeTemperatureSetpoints(
    input: Path,
    temperature_c: float = 22.0,
) -> None:
    dst = OUTPUT.get()
    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    ensure_scheduled_node_temperature_setpoints_inplace(epjson, temperature_c=temperature_c)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)


def ensure_scheduled_node_temperature_setpoints(
    epjson_in: Derivation, *, temperature_c: float = 22.0
) -> Derivation:
    return EnsureScheduledNodeTemperatureSetpoints(epjson_in, temperature_c=temperature_c)


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


def create_discovery_pipeline(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
) -> Derivation:
    """
    Create an epJSON suitable for *dummy simulations* whose purpose is to generate
    discovery artifacts such as `eplusout.edd` (actuator dictionary) and
    `eplustbl.htm` (tabular summary).

    Critical behavior:
    - Keep the building's default thermostat / load-based HVAC control intact.
      Some models rely on that to run successfully and to expose expected
      actuators in `.edd`.

    Downstream, you can convert this into a control-ready epJSON for RL by
    calling `create_control_pipeline(discovery_epjson)`.
    """
    current = upgrade(input_file, energyplus_path, src_version)
    current = convert_idf(current, energyplus_path)

    # Add meters and monitoring used by downstream parsers / diagnostics
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)
    current = add_edd_output(current)
    current = add_tabular_output(current)

    # Configure simulation output resolution
    current = modify_timestep(current, timesteps_per_hour=4)

    # IMPORTANT: include scheduled node setpoints (SetpointManager:Scheduled + Schedule:Constant)
    # already in the discovery model so the dummy simulation's `.edd` contains the
    # corresponding `Schedule:* / Schedule Value` actuators.
    #
    # This does NOT change HVAC control mode (we keep the building's thermostat/load control),
    # but it makes setpoint control robustly actuable downstream.
    current = ensure_scheduled_node_temperature_setpoints(current, temperature_c=22.0)

    # Make the name useful and explicit.
    current = Rename("building.discovery.epjson", current)
    return current


def create_control_pipeline(discovery_epjson: Derivation) -> Derivation:
    """
    Convert a discovery epJSON into a control-ready epJSON for the "airflow +
    System Node Setpoint" control mode.

    This is intentionally applied *after* discovery simulations, because those
    simulations rely on default thermostat/load control, while our RL control
    mode relies on `AirLoopHVAC:UnitarySystem.control_type = SetPoint`.
    """
    current = set_unitary_systems_to_setpoint_control(discovery_epjson)
    # Provide scheduled setpoints required by EnergyPlus for SetPoint control.
    # Implemented via SetpointManager:Scheduled + Schedule:Constant so control can
    # robustly override the schedule value at runtime.
    current = ensure_scheduled_node_temperature_setpoints(current, temperature_c=22.0)
    # Output variables to let us verify node temperatures + setpoints in `eplusout.csv/sql`.
    current = add_node_setpoint_diagnostics(current)
    # Ensure `.edd` generation is enabled (idempotent) for subsequent runs.
    current = add_edd_output(current)
    current = Rename("building.epjson", current)
    return current


def create_complete_pipeline(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str
    ) -> Derivation:
    """Create a complete processing pipeline from raw IDF to ready-to-go epJSON."""
    discovery = create_discovery_pipeline(input_file, energyplus_path, src_version)
    return create_control_pipeline(discovery)


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
    """
    Run an EnergyPlus simulation and save the output files.
    Use to run a dummy simulation
    """
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
    eio_file = tmp / "eplusout.eio"
    sql_file = tmp / "eplusout.sql"

    if not htm_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplustbl.htm")
    if not edd_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplusout.edd")
    if not eio_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplusout.eio")
    if not sql_file.exists():
        raise Exception("EnergyPlus simulation did not produce eplusout.sql")
    
    # Ensure output directory exists
    out.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(htm_file, out / "eplustbl.htm")
    shutil.copy(edd_file, out / "eplusout.edd")
    shutil.copy(eio_file, out / "eplusout.eio")
    shutil.copy(sql_file, out / "eplusout.sql")

###########################################################
# Analyse dummy simulation outputs
###########################################################

def eplustbl(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplustbl.htm file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplustbl.htm")


def eddfile(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplusout.edd file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplusout.edd")


def eiofile(ep_path: Path, epjson: Path, epw: Path) -> Derivation:
    """Get the eplusout.eio file from a simulation"""
    sim = run_simulation(ep_path, epjson, epw)
    return ChildFile(sim, "eplusout.eio")


def _parse_float_cell(x: str) -> float | None:
    s = str(x).strip()
    if not s or s.lower() in ("&nbsp;", "unknown"):
        return None
    try:
        return float(s)
    except Exception:
        return None


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
    - Coil speed/stage control (heating/cooling coils and unitary systems)
    - Fan air mass flow rate control
    - AirTerminal mass flow rate controls
    - AirLoopHVAC availability status override (force system on/off)
    - ZoneHVAC equipment actuators (e.g., PTAC, window AC, baseboards, etc.)
    
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
        "Coil Speed Control",
        "Fan Air Mass Flow Rate",
        # Air loop availability override (ForceOff / CycleOn / CycleOnZoneFansOnly)
        "AirLoopHVAC,Availability Status",
        # Zone equipment (cooling/heating terminals)
        "ZoneHVAC:",
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
            if "airterminal:" in line_lower:
                is_hvac = True
            elif ("fan," in line_lower and "mass flow" in line_lower):
                is_hvac = True
            elif "coil" in line_lower and ("speed" in line_lower or "stage" in line_lower):
                is_hvac = True
            elif "airloophvac," in line_lower and "availability status" in line_lower:
                is_hvac = True
            elif "zonehvac:" in line_lower:
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


def get_schedule_value_actuators(edd_path: Path) -> list[dict[str, str]]:
    """
    Extract schedule value actuators from an EnergyPlus `.edd` file.

    These actuators are typically of the form:
      EnergyManagementSystem:Actuator Available,<Schedule Name>,Schedule:*,Schedule Value,[ ]
    """
    out: list[dict[str, str]] = []
    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if "energymanagementsystem:actuator available" not in line.lower():
                continue
            if "schedule value" not in line.lower():
                continue

            parts = line.split(",", maxsplit=4)
            if len(parts) < 5:
                continue
            component_type = parts[2].strip()
            if not component_type.lower().startswith("schedule:"):
                continue

            out.append(
                {
                    "component_name": parts[1].strip(),
                    "component_type": component_type,
                    "control_type": parts[3].strip(),
                    "units": parts[4].strip(),
                }
            )
    return out


def get_b2b_scheduled_node_setpoint_actuators(
    edd_path: Path,
    *,
    schedule_names: Sequence[str],
) -> list[dict[str, str]]:
    """
    Return schedule-value actuators for the schedules used by `B2B Node Temp SPM ...`.

    These correspond to lines like:
      EnergyManagementSystem:Actuator Available,<Schedule Name>,Schedule:Constant,Schedule Value,[ ]
    """
    # EnergyPlus often uppercases object names in `.edd`, so match case-insensitively.
    want_upper = {str(s).strip().upper() for s in schedule_names if str(s).strip()}
    if not want_upper:
        return []

    out: list[dict[str, str]] = []
    for a in iter_edd_actuators(edd_path):
        ct = a.component_type.strip().lower()
        ctrl = a.control_type.strip().lower()
        if not ct.startswith("schedule:"):
            continue
        if ctrl != "schedule value":
            continue
        if a.component_name.strip().upper() not in want_upper:
            continue
        out.append(a.to_dict())

    return out


def get_baseboard_availability_schedule_names(epjson: dict[str, Any]) -> list[str]:
    """
    Return availability schedule names used by ZoneHVAC:Baseboard:Convective:Electric objects.
    """
    baseboards = epjson.get("ZoneHVAC:Baseboard:Convective:Electric", {})
    if not isinstance(baseboards, dict):
        return []
    names: list[str] = []
    for _bb_name, bb_obj_any in baseboards.items():
        if not isinstance(bb_obj_any, dict):
            continue
        sched = bb_obj_any.get("availability_schedule_name")
        if isinstance(sched, str) and sched.strip():
            names.append(sched.strip())
    return names


def get_zone_temperature_control_actuators(edd_path: Path) -> list[dict[str, str]]:
    """
    Extract Zone Temperature Control actuators (EMS overrides of thermostat setpoints).

    This is useful when we want to "force" EnergyPlus into heating/cooling mode without
    relying on the building's thermostat schedules.

    Args:
        edd_path: Path to an EnergyPlus `eplusout.edd` file.

    Returns:
        List of actuator descriptors for `get_actuator_handle()` with keys:
        'component_name', 'component_type', 'control_type', 'units'.
    """
    out: list[dict[str, str]] = []

    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if "zone temperature control" not in line.lower():
                continue
            # Typical `.edd` line format:
            # EnergyManagementSystem:Actuator Available,<Component Name>,<Component Type>,<Control Type>,<Units>
            parts = line.split(",", maxsplit=4)
            if len(parts) < 5:
                continue
            control_type = parts[3].strip()
            if control_type not in ("Heating Setpoint", "Cooling Setpoint"):
                continue
            out.append(
                {
                    "component_name": parts[1].strip(),
                    "component_type": parts[2].strip(),
                    "control_type": control_type,
                    "units": parts[4].strip(),
                }
            )

    return out


@dataclass(frozen=True, slots=True)
class EddActuatorDescriptor:
    """
    Strongly-typed representation of a single EMS actuator availability dictionary entry.

    This corresponds to `.edd` lines such as:
      EnergyManagementSystem:Actuator Available,<Component Name>,<Component Type>,<Control Type>,<Units>
    """

    component_name: str
    component_type: str
    control_type: str
    units: str

    @classmethod
    def from_edd_line(cls, line: str) -> "EddActuatorDescriptor | None":
        s = str(line).strip()
        if not s or s.startswith("!"):
            return None
        if "energymanagementsystem:actuator available" not in s.lower():
            return None

        parts = s.split(",", maxsplit=4)
        if len(parts) < 5:
            return None
        component_name = parts[1].strip()
        component_type = parts[2].strip()
        control_type = parts[3].strip()
        units = parts[4].strip()
        if not component_name or not component_type or not control_type:
            return None
        return cls(
            component_name=component_name,
            component_type=component_type,
            control_type=control_type,
            units=units,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "component_name": self.component_name,
            "component_type": self.component_type,
            "control_type": self.control_type,
            "units": self.units,
        }


def iter_edd_actuators(edd_path: Path) -> Iterator[EddActuatorDescriptor]:
    """
    Yield actuator descriptors from an EnergyPlus `.edd` actuator availability dictionary.
    """
    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            desc = EddActuatorDescriptor.from_edd_line(raw)
            if desc is not None:
                yield desc


def get_airflow_and_coil_node_setpoint_actuators(
    edd_path: Path,
    *,
    unitary_outlet_nodes: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """
    Extract actuators needed to control zone temperature via:

    - Fan air mass flow rate:
        <Fan Name>, Fan, Fan Air Mass Flow Rate, [kg/s]
    - Coil control via system node setpoints (Temperature Setpoint):
        <HEATING COIL NODE>, System Node Setpoint, Temperature Setpoint, [C]
        <SUPPLEMENTAL COIL NODE>, System Node Setpoint, Temperature Setpoint, [C]
        <COOLING COIL NODE>, System Node Setpoint, Temperature Setpoint, [C]
    - System availability override:
        <AirLoop Name>, AirLoopHVAC, Availability Status, [ ]

    This function is intentionally targeted (vs returning every System Node Setpoint),
    so action spaces stay compact and stable across buildings.
    """
    desired_node_suffixes = (
        "HEATING COIL NODE",
        "SUPPLEMENTAL COIL NODE",
        "COOLING COIL NODE",
    )

    def matches_any_suffix(node_name: str) -> bool:
        up = node_name.strip().upper()
        return any(sfx in up for sfx in desired_node_suffixes)

    outlet_nodes_norm: set[str] = set()
    if unitary_outlet_nodes is not None:
        for n in unitary_outlet_nodes:
            if isinstance(n, str) and n.strip():
                outlet_nodes_norm.add(n.strip().upper())

    selected: list[EddActuatorDescriptor] = []
    for a in iter_edd_actuators(edd_path):
        ct = a.component_type.strip().lower()
        ctrl = a.control_type.strip().lower()

        # 1) Fan air mass flow rate
        if ct == "fan" and ctrl == "fan air mass flow rate":
            selected.append(a)
            continue

        # 2) Coil node temperature setpoint actuators
        if ct == "system node setpoint" and ctrl == "temperature setpoint":
            if matches_any_suffix(a.component_name) or (
                outlet_nodes_norm and a.component_name.strip().upper() in outlet_nodes_norm
            ):
                selected.append(a)
            continue

        # 3) AirLoop availability override (force system available)
        if ct == "airloophvac" and ctrl == "availability status":
            selected.append(a)
            continue

    # Stable ordering + dedup
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for a in selected:
        key = f"{a.component_type}::{a.control_type}::{a.component_name}"
        if key in seen:
            continue
        seen.add(key)
        out.append(a.to_dict())
    return out