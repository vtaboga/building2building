import logging
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Union

import minergym.ontology as ontology
import numpy as np
from gymnasium.spaces import Box, Space
from minergym.simulation import ActuatorHole
from rdflib.term import Node

from .transform_utils import (
    Transform,
    TransformConcat,
    TransformList,
    TransformListToArray,
    TransformListToArrayShift,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualSetpoint:
    name: str
    heating_actuator: str
    heating_schedule: str
    cooling_actuator: str
    cooling_schedule: str


@dataclass(frozen=True)
class SingleHeating:
    name: str
    actuator: str
    schedule: str


@dataclass(frozen=True)
class SingleCooling:
    name: str
    actuator: str
    schedule: str


@dataclass(frozen=True)
class SingleHeatingOrCooling:
    name: str
    heating_actuator: str
    heating_schedule: str
    cooling_actuator: str
    cooling_schedule: str


ThermostatSetpoint = Union[
    DualSetpoint, SingleHeating, SingleCooling, SingleHeatingOrCooling
]


def get_controllable_setpoints(
    ont: ontology.Ontology,
) -> dict[str, list[ThermostatSetpoint]]:
    """
    Analyzes an RDF representation of an EnergyPlus model to identify zone temperature
    setpoints that can be overwritten with the Python/EnergyPlus API.
    """
    # Initialize result dictionary
    zone_setpoints: dict[str, list[ThermostatSetpoint]] = {}

    # Step 1: Find all zones

    zones = [str(zone) for zone in ont.zones()]
    logger.debug(f"Found zones: {zones}")

    # Step 2: Find all zone thermostat controls
    zone_control_query = """
    SELECT ?control_name ?zone_name
    WHERE {
        ?control_name a "ZoneControl:Thermostat" .
        ?control_name idf:zone_or_zonelist_name ?zone_name .
    }
    """

    zone_thermostats = {}
    thermostat_zones = {}

    # Execute the query to get zone-thermostat relationships
    logger.info("Querying for thermostats...")
    for row in ont.rdf.query(zone_control_query):
        control = row.control_name

        # Strip URI from zone name
        zone_or_list = row.zone_name
        logger.debug(f"Found thermostat control: {control} for zone: {zone_or_list}")

        # Check if zone is a ZoneList
        zone_list_query = """
        SELECT ?zone
        WHERE {
        ?zone_list a "ZoneList" .
        ?zone_list idf:zones ?zones .
        ?zones rdf:rest*/rdf:first ?zone .
        }
        """

        zone_list_results = ont.rdf.query(
            zone_list_query, initBindings={"zone_list": zone_or_list}
        )
        zones_in_list = [row.zone for row in zone_list_results]
        # If no zones found in list, it's a direct zone reference
        if not zones_in_list:
            logger.debug("`zone_or_list` is a direct zone reference")
            zones_in_list = [zone_or_list]

        # Add to mappings
        for zone in zones_in_list:
            zone_thermostats[zone] = control

        thermostat_zones[control] = zones_in_list

    logger.debug(f"found the zones of each thermostat: {thermostat_zones}")

    # Step 3: For each thermostat, find the setpoint object
    logger.info("Looking for setpoint objects...")
    for thermostat, zones in thermostat_zones.items():
        logger.debug(f"Checking thermostat: {thermostat}")
        # Get the thermostat setpoint type and name

        setpoint_query = """
        SELECT ?control_type ?setpoint
        WHERE {
        ?thermostat a "ZoneControl:Thermostat" .
        ?thermostat idf:control_1_object_type ?control_type .
        ?thermostat idf:control_1_name ?setpoint
        }
        """

        logger.debug("Running setpoint query...")
        setpoint_results = list(
            ont.rdf.query(setpoint_query, initBindings={"thermostat": thermostat})
        )

        # setpoint_results = list(rdf_graph.query(setpoint_query, initNs={"ns": ns}))
        logger.debug(f"Found {len(setpoint_results)} setpoint results")

        if not setpoint_results:
            logger.debug(f"Couldn't find setpoint for thermostat {thermostat}")
            continue

        control_type: Node = setpoint_results[0].control_type
        setpoint: Node = setpoint_results[0].setpoint
        logger.debug(f"Control type: {control_type}")
        logger.debug(f"Setpoint name: {setpoint}")

        # Handle different thermostat setpoint types
        for zone in zones:
            zone_name = str(zone)

            if zone_name not in zone_setpoints:
                zone_setpoints[zone_name] = []

            if str(control_type) == "ThermostatSetpoint:DualSetpoint":
                logger.debug(f"Querying dual setpoint schedules for: {setpoint}")
                # Get heating and cooling schedule names
                dual_query = """
                SELECT ?heating_schedule ?cooling_schedule
                WHERE {
                ?setpoint a "ThermostatSetpoint:DualSetpoint" .
                ?setpoint idf:heating_setpoint_temperature_schedule_name ?heating_schedule .
                ?setpoint idf:cooling_setpoint_temperature_schedule_name ?cooling_schedule .
                }
                """

                # urllib.parse.quote(setpoint_name)

                logger.debug("Running dual setpoint query...")
                dual_results = list(
                    ont.rdf.query(dual_query, initBindings={"setpoint": setpoint})
                )
                logger.debug(f"Found {len(dual_results)} schedule results")

                for row in dual_results:
                    heating_schedule = str(row.heating_schedule)
                    cooling_schedule = str(row.cooling_schedule)
                    logger.debug(f"Heating schedule: {heating_schedule}")
                    logger.debug(f"Cooling schedule: {cooling_schedule}")

                    # Add heating setpoint with decoded zone name
                    decoded_zone = str(zone)

                    zone_setpoints[zone_name].append(
                        DualSetpoint(
                            name=setpoint.toPython(),
                            heating_actuator=f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                            heating_schedule=heating_schedule,
                            cooling_actuator=f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                            cooling_schedule=cooling_schedule,
                        )
                    )

            elif control_type == "ThermostatSetpoint:SingleHeating":
                # Get heating schedule name

                heating_query = """
                SELECT ?schedule
                WHERE {
                ?setpoint a "ThermostatSetpoint:SingleHeating" .
                ?setpoint idf:setpoint_temperature_schedule_name ?schedule .
                }
                """

                for row in ont.rdf.query(
                    heating_query, initBindings={"setpoint": setpoint}
                ):
                    schedule = str(row.schedule)
                    decoded_zone = str(zone)

                    zone_setpoints[zone_name].append(
                        SingleHeating(
                            name=setpoint.toPython(),
                            actuator=f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                            schedule=schedule,
                        )
                    )

            elif control_type == "ThermostatSetpoint:SingleCooling":
                # Get cooling schedule name

                cooling_query = """
                SELECT ?schedule
                WHERE {
                ?setpoint a "ThermostatSetpoint:SingleCooling" .
                ?setpoint idf:setpoint_temperature_schedule_name ?schedule .
                }"""

                for row in ont.rdf.query(
                    cooling_query, initBindings={"setpoint": setpoint}
                ):
                    schedule = str(row.schedule)
                    decoded_zone = urllib.parse.unquote(zone)

                    zone_setpoints[zone_name].append(
                        SingleCooling(
                            name=setpoint.toPython(),
                            actuator=f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                            schedule=schedule,
                        )
                    )

            elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
                # Get schedule name

                schedule_query = """
                SELECT ?schedule
                WHERE {
                ?setpoint a "ThermostatSetpoint:SingleHeatingOrCooling" .
                ?setpoint idf:setpoint_temperature_schedule_name ?schedule .
                }
                """

                for row in ont.rdf.query(
                    schedule_query, initBindings={"setpoint": setpoint}
                ):
                    schedule = str(row.schedule)
                    decoded_zone = str(zone)

                    # This type can switch between heating and cooling, so create both actuator keys
                    zone_setpoints[zone_name].append(
                        SingleHeatingOrCooling(
                            name=setpoint.toPython(),
                            heating_actuator=f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                            heating_schedule=schedule,
                            cooling_actuator=f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                            cooling_schedule=schedule,
                        )
                    )
            else:
                logger.debug(f"found strange control type: {control_type}")

    return zone_setpoints


def single_thermostat_transform(t: ThermostatSetpoint) -> Transform[list, Box]:
    match t:
        case DualSetpoint():
            return TransformListToArrayShift(
                [
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.heating_schedule
                    ),
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.cooling_schedule
                    ),
                ],
                Box(
                    np.array([15.0, 1.0]),
                    np.array([25.0, 15.0]),
                ),
            )

        case SingleHeating():
            return TransformListToArray(
                [ActuatorHole("Schedule:Compact", "Schedule Value", t.schedule)],
                Box(np.array([15]), np.array([25])),
            )

        case SingleCooling():
            return TransformListToArray(
                ActuatorHole("Schedule:Compact", "Schedule Value", t.schedule),
                Box(np.array([16]), np.array([40])),
            )

        case SingleHeatingOrCooling():
            return TransformListToArrayShift(
                [
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.heating_schedule
                    ),
                    ActuatorHole(
                        "Schedule:Compact", "Schedule Value", t.cooling_schedule
                    ),
                ],
                Box(
                    np.array([15.0, 1.0]),
                    np.array([25.0, 15.0]),
                ),
            )

        case _:
            raise Exception(f"Should be unreachable. Got a {t}")


def many_thermostats_transform(
    setpoints: list[ThermostatSetpoint],
) -> Transform[list, Box]:
    return TransformConcat(
        TransformList([single_thermostat_transform(sp) for sp in setpoints])
    )
