# Here all the steps necessary to transform a raw idf into a ready-to-go idf are
# expressed as BuildStep's so that they can be properly cached.
import contextlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from building2building.store import BaseDerivation, Child, Derivation, default_hash

logger = logging.getLogger(__name__)

Transition: TypeAlias = Literal[
    "9.4.0-to-9.5.0",
    "9.5.0-to-9.6.0",
    "9.6.0-to-22.1.0",
    "22.1.0-to-22.2.0",
    "22.2.0-to-23.1.0",
    "23.1.0-to-23.2.0",
    "23.2.0-to-24.1.0",
]


@dataclass
class UpgradeIDF(BaseDerivation):
    input: Derivation
    transition_exe: Derivation
    from_idd: Derivation
    to_idd: Derivation

    def name(self) -> str:
        p = Path(self.input.name())
        return p.with_stem(p.stem + "-plus").name

    def build(self, dst: Path, deps):
        with tempfile.TemporaryDirectory() as tempdir:
            with contextlib.chdir(tempdir):
                Path(self.from_idd.name()).symlink_to(deps["from_idd"])
                Path(self.to_idd.name()).symlink_to(deps["to_idd"])
                shutil.copy(deps["input"], "in.idf")

                temp_idf_out = Path(tempdir) / "in.idfnew"
                # Run the transition executable from the temp_dir
                cmdline = [
                    str(deps["transition_exe"]),
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
                raise Exception(f"{temp_idf_out} does not exist")

            shutil.copy(temp_idf_out, dst)


def upgrade_idf(
    idf_in: Derivation, transition: Transition, energyplus_path: Derivation
) -> Derivation:
    from_version, to_version = transition.split("-to-")

    transition_dir = Child(energyplus_path, "PreProcess/IDFVersionUpdater")

    transition_exe = Child(
        transition_dir,
        f"Transition-V{from_version.replace('.', '-')}-to-V{to_version.replace('.', '-')}",
    )

    from_idd_part = f"V{from_version.replace('.', '-')}-Energy+.idd"
    to_idd_part = f"V{to_version.replace('.', '-')}-Energy+.idd"

    from_idd = Child(transition_dir, from_idd_part)
    to_idd = Child(transition_dir, to_idd_part)

    return UpgradeIDF(
        idf_in,
        transition_exe,
        from_idd,
        to_idd,
    )


@dataclass
class ConvertIDF(BaseDerivation):
    input: Derivation
    converter: Derivation

    def name(self) -> str:
        p = Path(self.input.name())
        return p.with_suffix(".epJSON").name

    def build(self, dst: Path, deps):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            temp_idf_path = temp_path / "in.idf"
            temp_epjson_path = temp_idf_path.with_suffix(".epJSON")

            shutil.copy(deps["input"], temp_idf_path)

            result = subprocess.run(
                [str(deps["converter"]), str(temp_idf_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp,
            )

            if result.stderr:
                logger.warning(f"Conversion warnings: {result.stderr}")
            if result.stdout:
                logger.warning(f"Conversion output: {result.stdout}")

            shutil.copy(temp_epjson_path, dst)


@dataclass
class AddHVACMeters(BaseDerivation):
    input: Derivation

    def name(self) -> str:
        p = Path(self.input.name())
        return p.with_stem(p.stem + "-hvac-meters").name

    def build(self, dst: Path, deps):
        with open(deps["input"], "r") as f:
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


@dataclass
class AddOutdoorAirMeters(BaseDerivation):
    input: Derivation

    def name(self) -> str:
        p = Path(self.input.name())
        return p.with_stem(p.stem + "-outdoor-air").name

    def build(self, dst: Path, deps):
        with open(deps["input"], "r") as f:
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


@dataclass
class ModifyTimestep(BaseDerivation):
    input: Derivation
    timesteps_per_hour: int = 4

    def name(self) -> str:
        p = Path(self.input.name())
        return p.with_stem(p.stem + f"-timestep{self.timesteps_per_hour}").name

    def build(self, dst: Path, deps):
        with open(deps["input"], "r") as f:
            epjson = json.load(f)

        # Check for existing Timestep object
        timestep_modified = False

        if "Timestep" in epjson:
            # Process all timestep objects (usually there's just one)
            for timestep_key, timestep_data in epjson["Timestep"].items():
                current_timestep = timestep_data.get("number_of_timesteps_per_hour", 1)
                if current_timestep != self.timesteps_per_hour:
                    logger.info(
                        f"Changing timestep from {current_timestep} to {self.timesteps_per_hour}"
                    )
                    epjson["Timestep"][timestep_key]["number_of_timesteps_per_hour"] = (
                        self.timesteps_per_hour
                    )
                    timestep_modified = True
                else:
                    logger.info(f"Timestep already set to {self.timesteps_per_hour}")
        else:
            # Create new Timestep object
            logger.info(
                f"Creating Timestep object with {self.timesteps_per_hour} timesteps per hour"
            )
            epjson["Timestep"] = {
                "Timestep 1": {"number_of_timesteps_per_hour": self.timesteps_per_hour}
            }
            timestep_modified = True

        with open(dst, "w") as f:
            json.dump(epjson, f, indent=4)


@dataclass
class AddSetpointControl(BaseDerivation):
    input: Derivation

    def name(self) -> str:
        p = Path(self.input.name())
        return p.with_stem(p.stem + "-setpoint-control").name

    def build(self, dst: Path, deps):
        with open(deps["input"], "r") as f:
            epjson = json.load(f)

        # Get thermostat setpoints used in the building
        thermostat_setpoints = self._get_temperature_setpoints(epjson)

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

    def _get_temperature_setpoints(self, epjson_data) -> list[tuple]:
        """Find thermostats that control zones."""
        results = []

        # Look for zone controls that reference thermostat setpoints
        if "ZoneControl:Thermostat" in epjson_data:
            for control_name, control_data in epjson_data[
                "ZoneControl:Thermostat"
            ].items():
                control_type = control_data.get("control_1_object_type")
                setpoint_name = control_data.get("control_1_name")

                if control_type and setpoint_name:
                    results.append((control_type, setpoint_name))

        return results


@dataclass
class GlueSurfaces(BaseDerivation):
    input: Derivation

    def name(self) -> str:
        p = Path(self.input.name())
        return p.with_stem(p.stem + "-glued").name

    def build(self, dst: Path, deps):
        with open(deps["input"], "r") as f:
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


def convert_idf(idf_in: Derivation, energyplus_path: Derivation) -> Derivation:
    """Create an IDF to epJSON conversion build step."""
    converter = Child(energyplus_path, "ConvertInputFormat")
    return ConvertIDF(idf_in, converter)


def add_hvac_meters(epjson_in: Derivation) -> Derivation:
    """Add HVAC energy meters to epJSON."""
    return AddHVACMeters(epjson_in)


def add_outdoor_air_meters(epjson_in: Derivation) -> Derivation:
    """Add outdoor air monitoring to epJSON."""
    return AddOutdoorAirMeters(epjson_in)


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


def upgrade(
    input_file: Derivation, energyplus_path: Derivation, transitions=all_transitions
) -> Derivation:
    # Multi-step upgrade process

    # Chain upgrades
    current = input_file
    for transition in transitions:
        current = upgrade_idf(current, transition, energyplus_path)

    return current


def create_complete_pipeline(
    input_file: Derivation, energyplus_path: Derivation, *, transitions=all_transitions
) -> Derivation:
    """Create a complete processing pipeline from raw IDF to ready-to-go epJSON."""

    current = upgrade(input_file, energyplus_path, transitions=transitions)

    # Convert to epJSON
    current = convert_idf(current, energyplus_path)

    # Add meters and monitoring
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)

    # Configure simulation
    current = modify_timestep(current, timesteps_per_hour=4)
    current = add_setpoint_control(current)

    # # Final processing
    # current = glue_surfaces(current)

    return current
