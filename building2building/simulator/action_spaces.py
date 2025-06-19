import numpy as np
import gymnasium as gym
import logging
import urllib.parse
import rdflib
from rdflib import namespace
from rdflib.term import Node
from building2building.core.logging import setup_logger
from typing import reveal_type, Dict, List
import building2building.ontology as ontology


logger = logging.getLogger(__name__)


def action_transform(act, actuators):
    """Transform raw actions into heating and cooling setpoints.
    
    Args:
        act: Raw actions from the policy
        actuators: Dictionary of actuators with their schedule names
    """
    # Get schedule names from actuators
    schedule_names = list(actuators.keys())
    
    # Ensure we have exactly two schedules (heating and cooling)
    if len(schedule_names) != 2:
        raise ValueError(f"Expected 2 schedule names (heating and cooling), got {len(schedule_names)}")
    
    # Sort to ensure consistent ordering (heating should be first due to alphabetical order)
    schedule_names.sort()

    heating_setpoint = round(act[0], 1)
    offset = round(act[1], 1)
    
    return {
        schedule_names[0]: heating_setpoint + offset,  # Cooling setpoint (first alphabetically)
        schedule_names[1]: heating_setpoint,  # Heating setpoint (second alphabetically)
    }

def create_action_space(actuators) -> gym.spaces.Box:
    """Create a Gymnasium action space for temperature setpoints."""
    # Verify we have exactly two actuators (heating and cooling)
    if len(actuators) != 2:
        raise ValueError(f"Expected 2 actuators (heating and cooling), got {len(actuators)}")
    
    # Temperature bounds 
    # The first valus is the heating setpoint
    # The second value is the offset from the heating setpoint to the cooling setpoint: T_cooling = T_heating + offset
    return gym.spaces.Box(
        np.array([15.0, 1.0]),  
        np.array([25.0, 15.0])
    )


def get_controllable_setpoints(ont: ontology.Ontology) -> Dict[str, List[Dict]]:
    """
    Analyzes an RDF representation of an EnergyPlus model to identify zone temperature 
    setpoints that can be overwritten with the Python/EnergyPlus API.

    Args:
        rdf_graph (rdflib.Graph): RDF graph representation of an epJSON file

    Returns:
        Dict[str, List[Dict]]: Dictionary mapping zones to their controllable setpoints
            {
                "zone_name": [
                    {
                        "setpoint_type": "heating" or "cooling", 
                        "actuator_key": "key to use with API",
                        "schedule_name": "original schedule name",
                        "control_type": "type of thermostat control"
                    }
                ]
            }
    """
    # Initialize result dictionary
    zone_setpoints = {}

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

        zone_list_results = ont.rdf.query(zone_list_query, initBindings={"zone_list": zone_or_list})
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
        setpoint_results = list(ont.rdf.query(setpoint_query, initBindings={"thermostat": thermostat}))

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
                dual_results = list(ont.rdf.query(dual_query, initBindings={"setpoint": setpoint}))
                logger.debug(f"Found {len(dual_results)} schedule results")



                for row in dual_results:
                    heating_schedule = str(row.heating_schedule)
                    cooling_schedule = str(row.cooling_schedule)
                    logger.debug(f"Heating schedule: {heating_schedule}")
                    logger.debug(f"Cooling schedule: {cooling_schedule}")

                    # Add heating setpoint with decoded zone name
                    decoded_zone = str(zone)
                    zone_setpoints[zone_name].append({
                        "setpoint_type": "heating",
                        "actuator_key": f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                        "schedule_name": heating_schedule,
                        "control_type": "DualSetpoint"
                    })

                    # Add cooling setpoint
                    zone_setpoints[zone_name].append({
                        "setpoint_type": "cooling",
                        "actuator_key": f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                        "schedule_name": cooling_schedule,
                        "control_type": "DualSetpoint"
                    })

            elif control_type == "ThermostatSetpoint:SingleHeating":
                # Get heating schedule name

                heating_query = """
                SELECT ?schedule
                WHERE {
                ?setpoint a "ThermostatSetpoint:SingleHeating" .
                ?setpoint idf:setpoint_temperature_schedule_name ?schedule .
                }
                """

                for row in ont.rdf.query(heating_query, initBindings={"setpoint": setpoint}):
                    schedule = str(row.schedule)
                    decoded_zone = str(zone)

                    zone_setpoints[zone_name].append({
                        "setpoint_type": "heating",
                        "actuator_key": f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleHeating"
                    })

            elif control_type == "ThermostatSetpoint:SingleCooling":
                # Get cooling schedule name

                cooling_query = """
                SELECT ?schedule
                WHERE {
                ?setpoint a "ThermostatSetpoint:SingleCooling" .
                ?setpoint idf:setpoint_temperature_schedule_name ?schedule .
                }"""

                for row in ont.rdf.query(cooling_query, initBindings={"setpoint": setpoint}):
                    schedule = str(row.schedule)
                    decoded_zone = urllib.parse.unquote(zone)

                    zone_setpoints[zone_name].append({
                        "setpoint_type": "cooling",
                        "actuator_key": f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleCooling"
                    })

            elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
                # Get schedule name

                schedule_query = """
                SELECT ?schedule
                WHERE {
                ?setpoint a "ThermostatSetpoint:SingleHeatingOrCooling" .
                ?setpoint idf:setpoint_temperature_schedule_name ?schedule .
                }
                """

                for row in ont.rdf.query(schedule_query, initBindings={"setpoint": setpoint}):
                    schedule = str(row.schedule)
                    decoded_zone = str(zone)

                    # This type can switch between heating and cooling, so create both actuator keys
                    zone_setpoints[zone_name].append({
                        "setpoint_type": "heating",
                        "actuator_key": f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleHeatingOrCooling"
                    })

                    zone_setpoints[zone_name].append({
                        "setpoint_type": "cooling",
                        "actuator_key": f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleHeatingOrCooling"
                    })

    return zone_setpoints
