"""To run an EnergyPlus simulation and wrap it into the neat

observation -> policy -> action -> environment -> observation

box commonly used in RL, it is necessary to have the following components:

1. A building file.
2. A weather file.
3. The set of EP variables which will be written to the observations.
4. The set of EP actuators which will be read from the action.

This module collects functions useful to creates those 4-tuples.

"""

import simulator.query_info as query_info
import rdflib
import typing
import simulator.simulation as simulation


def auto_get_actuators(
    rdf: rdflib.Graph,
) -> typing.Dict[str, simulation.ActuatorHole]:
    """Add all actuators listed in the graph. This is probably not what you
    want, since actuators that are not heating/cooling setpoints will be added
    too."""
    act = {}
    for name in query_info.rdf_schedules(rdf):
        # for name in zones_with_cooling
        act[name] = simulation.ActuatorHole("Schedule:Compact", "Schedule Value", name)
    return act


def auto_add_temperature(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
) -> None:
    """Add a "ZONE AIR TEMPERATURE" for each zone in the graph."""
    temps = {}
    temps["environment"] = simulation.VariableHole(
        "SITE OUTDOOR AIR DRYBULB TEMPERATURE",
        "ENVIRONMENT",
    )
    for z in query_info.rdf_zones(rdf):
        temps[z] = simulation.VariableHole("ZONE AIR TEMPERATURE", z)
        obs_template["temperature"] = temps


def auto_add_setpoint_variables(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
) -> None:
    setpoints: typing.Any = {}
    obs_template["setpoints"] = setpoints

    heating: typing.Any = {}
    setpoints["heating"] = heating

    cooling: typing.Any = {}
    setpoints["cooling"] = cooling

    for z in query_info.rdf_zones(rdf):
        heating[z] = simulation.VariableHole(
            "Zone Thermostat Heating Setpoint Temperature", z
        )
        cooling[z] = simulation.VariableHole(
            "Zone Thermostat Cooling Setpoint Temperature", z
        )


def auto_add_comfort(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
) -> None:
    if "comfort" not in obs_template:
        obs_template["comfort"] = {}

    comfort = obs_template["comfort"]
    for z in query_info.rdf_zones(rdf):
        comfort[z + "_comfort"] = simulation.VariableHole(
            "Zone Thermal Comfort Pierce Model Thermal Sensation Index", z
        )
        comfort[z + "_discomfort"] = simulation.VariableHole(
            "Zone Thermal Comfort Pierce Model Discomfort Index", z
        )


def auto_add_energy(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
) -> None:
    if "reward" not in obs_template:
        obs_template["energy"] = {}

    r = obs_template["energy"]

    r["whole_building"] = simulation.MeterHole("Electricity:HVAC")

    for z in query_info.rdf_zones(rdf):
        r[z + "_cooling"] = simulation.VariableHole(
            "Zone Air System Sensible Cooling Energy", z
        )
        r[z + "_heating"] = simulation.VariableHole(
            "Zone Air System Sensible Heating Energy", z
        )


def auto_add_time(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
) -> None:
    """Add ubiquitous variables."""

    time: typing.Any = {}
    obs_template["time"] = time
    time["current_time"] = simulation.FunctionHole(simulation.api.exchange.current_time)
    time["day_of_year"] = simulation.FunctionHole(simulation.api.exchange.day_of_year)

    # # All of those don't work
    # time["current_sim_time"] = myeplus.Function(myeplus.api.exchange.current_sim_time)
    # time["day_of_month"] = myeplus.Function(myeplus.api.exchange.day_of_month)
    # time["day_of_week"] = myeplus.Function(myeplus.api.exchange.day_of_week)
    # time["actual_date_time"] = myeplus.Function(myeplus.api.exchange.actual_date_time)
    # time["year"] = myeplus.Function(myeplus.api.exchange.year)
