"""To run an EnergyPlus simulation and wrap it into the neat

observation -> policy -> action -> environment -> observation

box commonly used in RL, it is necessary to have the following components:

1. A building file.
2. A weather file.
3. The set of EP variables which will be written to the observations.
4. The set of EP actuators which will be read from the action.

This module collects functions useful to creates those 4-tuples.

"""

import building2building.ontology as ontology
import rdflib
import building2building.simulator.simulation as simulation
from typing import Dict, List, reveal_type, Any
import urllib.parse
import logging
import os


def auto_get_actuators(
    ont: ontology.Ontology,
) -> dict[str, simulation.ActuatorHole]:
    """Add all actuators listed in the graph. This is probably not what you
    want, since actuators that are not heating/cooling setpoints will be added
    too."""
    act = {}
    for sch in ont.schedules():
        # for name in zones_with_cooling
        act[str(sch)] = simulation.ActuatorHole("Schedule:Compact", "Schedule Value", str(sch))
    return act


def auto_add_setpoint_variables(
    ont: ontology.Ontology, obs_template: dict[str, Any]
) -> None:
    setpoints: Any = {}
    obs_template["setpoints"] = setpoints

    heating: Any = {}
    setpoints["heating"] = heating

    cooling: Any = {}
    setpoints["cooling"] = cooling

    for z in ont.zones():
        # URL-decode the zone name and use it as the key
        decoded_zone = str(z)
        heating[decoded_zone] = simulation.VariableHole(
            "Zone Thermostat Heating Setpoint Temperature", decoded_zone
        )
        cooling[decoded_zone] = simulation.VariableHole(
            "Zone Thermostat Cooling Setpoint Temperature", decoded_zone
        )


def auto_add_temperature(
        ont: ontology.Ontology, obs_template: dict[str, Any]
) -> None:
    """Add zone air temperatures to the observation template."""
    if "temperature" not in obs_template:
        obs_template["temperature"] = {}

    temps = obs_template["temperature"]
    for z in ont.zones():
        # URL-decode the zone name and use it as the key
        decoded_zone = str(z)
        temps[decoded_zone] = simulation.VariableHole("ZONE AIR TEMPERATURE", decoded_zone)

def auto_add_energy(
    ont: ontology.Ontology, obs_template: dict[str, Any]
) -> None:
    """Add HVAC energy consumption meters to the observation template."""
    if "energy" not in obs_template:
        obs_template["energy"] = {}

    energy = obs_template["energy"]
    # Add whole building HVAC energy meters only
    energy["HVAC_electricity"] = simulation.MeterHole("Electricity:HVAC")
    energy["HVAC_natural_gas"] = simulation.MeterHole("NaturalGas:HVAC")


def auto_add_time(
    ont: ontology.Ontology, obs_template: dict[str, Any]
) -> None:
    """Add time variables to the observation template."""
    if "time" not in obs_template:
        obs_template["time"] = {}

    time = obs_template["time"]
    time["current_time"] = simulation.FunctionHole(simulation.api.exchange.current_time)
    time["day_of_year"] = simulation.FunctionHole(simulation.api.exchange.day_of_year)

    # # All of those don't work
    # time["current_sim_time"] = myeplus.Function(myeplus.api.exchange.current_sim_time)
    # time["day_of_month"] = myeplus.Function(myeplus.api.exchange.day_of_month)
    # time["day_of_week"] = myeplus.Function(myeplus.api.exchange.day_of_week)
    # time["actual_date_time"] = myeplus.Function(myeplus.api.exchange.actual_date_time)
    # time["year"] = myeplus.Function(myeplus.api.exchange.year)


def auto_add_weather(
    ont: ontology.Ontology, obs_template: dict[str, Any]
) -> None:
    """Add outdoor air measurements to the observation template."""
    if "weather" not in obs_template:
        obs_template["weather"] = {}

    weather = obs_template["weather"]
    weather["drybulb_temp"] = simulation.VariableHole(
        "Site Outdoor Air Drybulb Temperature", "Environment"
    )
    weather["relative_humidity"] = simulation.VariableHole(
        "Site Outdoor Air Relative Humidity", "Environment"
    )

