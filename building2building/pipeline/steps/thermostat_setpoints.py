import json
from pathlib import Path

from building2building.store import Derivation, OUTPUT, derivation


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
                epjson["Schedule:Compact"][cooling_schedule_name] = create_schedule_compact(
                    40.0
                )

            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = create_schedule_compact(
                    10.0
                )

        elif control_type == "ThermostatSetpoint:SingleHeating":
            heating_schedule_name = f"{setpoint_name} Heating Setpoint"
            epjson["ThermostatSetpoint:SingleHeating"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = heating_schedule_name

            if heating_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][heating_schedule_name] = create_schedule_compact(
                    10.0
                )

        elif control_type == "ThermostatSetpoint:SingleCooling":
            cooling_schedule_name = f"{setpoint_name} Cooling Setpoint"
            epjson["ThermostatSetpoint:SingleCooling"][setpoint_name][
                "setpoint_temperature_schedule_name"
            ] = cooling_schedule_name

            if cooling_schedule_name not in epjson["Schedule:Compact"]:
                epjson["Schedule:Compact"][cooling_schedule_name] = create_schedule_compact(
                    40.0
                )

        elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
            raise NotImplementedError(
                "ThermostatSetpoint:SingleHeatingOrCooling not supported yet"
            )

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


def get_temperature_setpoints(epjson_data) -> list[tuple]:
    """Find thermostats that control zones."""
    results = []

    # Look for zone controls that reference thermostat setpoints
    if "ZoneControl:Thermostat" in epjson_data:
        for _control_name, control_data in epjson_data["ZoneControl:Thermostat"].items():
            control_type = control_data.get("control_1_object_type")
            setpoint_name = control_data.get("control_1_name")

            if control_type and setpoint_name:
                results.append((control_type, setpoint_name))

    return results


def add_setpoint_control(epjson_in: Derivation) -> Derivation:
    """Add controllable temperature setpoints."""
    return AddSetpointControl(epjson_in)

