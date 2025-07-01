import numpy as np
import typing
import gymnasium as gym
import logging
import urllib.parse
import rdflib
from rdflib import namespace
from typing import Dict, List
from building2building.simulator.query_info import ns

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


def get_controllable_setpoints_rdf(rdf_graph: rdflib.Graph) -> Dict[str, List[Dict]]:
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
    
    isa = rdflib.namespace.RDF.type
    
    # Step 1: Find all zones
    zones_query = """
    SELECT ?zone_name
    WHERE {
        ?zone_name a ns:Zone .
    }
    """
    
    zones = [str(row.zone_name) for row in rdf_graph.query(zones_query, initNs={"ns": ns})]
    logger.debug(f"Found zones: {zones}")
    
    # Step 2: Find all zone thermostat controls
    zone_control_query = """
    SELECT ?control_name ?zone_name
    WHERE {
        ?control_name a ns:ZoneControl%3AThermostat .
        ?control_name ?zone_prop ?zone_name .
        FILTER(?zone_prop = ns:zone_or_zonelist_name) .
    }
    """
    
    zone_thermostats = {}
    thermostat_zones = {}
    
    # Execute the query to get zone-thermostat relationships
    logger.info("Querying for thermostats...")
    for row in rdf_graph.query(zone_control_query, initNs={"ns": ns}):
        control_name = str(row.control_name)
        # Strip URI from zone name
        zone_name = str(row.zone_name).replace(str(ns), '')
        logger.debug(f"Found thermostat control: {control_name} for zone: {zone_name}")
        
        # Check if zone_name is a ZoneList
        zone_list_query = f"""
        SELECT ?zone
        WHERE {{
            ?list_name a ?type .
            FILTER(?type = "ZoneList") .
            FILTER(?list_name = "{zone_name}"^^<http://www.w3.org/2001/XMLSchema#string>) .
            ?list_name ?zones_prop ?zones .
            FILTER(?zones_prop = ns:zones) .
            ?zones ?member ?zone .
        }}
        """
        
        zone_list_results = rdf_graph.query(zone_list_query, initNs={"ns": ns})
        zones_in_list = [str(row.zone) for row in zone_list_results]
        # If no zones found in list, it's a direct zone reference
        if not zones_in_list:
            zones_in_list = [zone_name]
        
        # Add to mappings
        for zone in zones_in_list:
            zone_thermostats[zone] = control_name
            
        thermostat_zones[control_name] = zones_in_list
    
    # Step 3: For each thermostat, find the setpoint object
    logger.info("Looking for setpoint objects...")
    for thermostat, zones in thermostat_zones.items():
        logger.debug(f"Checking thermostat: {thermostat}")
        # Get the thermostat setpoint type and name
        setpoint_query = f"""
        SELECT ?control_type ?setpoint_name
        WHERE {{
            ?control a ns:ZoneControl%3AThermostat .
            FILTER(?control = <{thermostat}>) .
            ?control ?control_type_prop ?control_type .
            FILTER(?control_type_prop = ns:control_1_object_type) .
            ?control ?control_name_prop ?setpoint_name .
            FILTER(?control_name_prop = ns:control_1_name) .
        }}
        """
        
        logger.debug("Running setpoint query...")
        setpoint_results = list(rdf_graph.query(setpoint_query, initNs={"ns": ns}))
        logger.debug(f"Found {len(setpoint_results)} setpoint results")
        
        if not setpoint_results:
            continue
            
        control_type = str(setpoint_results[0].control_type)
        setpoint_name = str(setpoint_results[0].setpoint_name)
        logger.debug(f"Control type: {control_type}")
        logger.debug(f"Setpoint name: {setpoint_name}")
        
        # Handle different thermostat setpoint types
        for zone in zones:
            if zone not in zone_setpoints:
                zone_setpoints[zone] = []
                
            if control_type == "ThermostatSetpoint:DualSetpoint":
                logger.debug(f"Querying dual setpoint schedules for: {setpoint_name}")
                # Get heating and cooling schedule names
                dual_query = f"""
                SELECT ?heating_schedule ?cooling_schedule
                WHERE {{
                    ?setpoint a ns:ThermostatSetpoint%3ADualSetpoint .
                    BIND(ns:{urllib.parse.quote(setpoint_name)} AS ?expected_name)
                    FILTER(?setpoint = ?expected_name) .
                    ?setpoint ?heating_prop ?heating_schedule .
                    FILTER(?heating_prop = ns:heating_setpoint_temperature_schedule_name) .
                    ?setpoint ?cooling_prop ?cooling_schedule .
                    FILTER(?cooling_prop = ns:cooling_setpoint_temperature_schedule_name) .
                }}
                """
                logger.debug("Running dual setpoint query...")
                dual_results = list(rdf_graph.query(dual_query, initNs={"ns": ns}))
                logger.debug(f"Found {len(dual_results)} schedule results")
                
                for row in dual_results:
                    heating_schedule = str(row.heating_schedule)
                    cooling_schedule = str(row.cooling_schedule)
                    logger.debug(f"Heating schedule: {heating_schedule}")
                    logger.debug(f"Cooling schedule: {cooling_schedule}")
                    
                    # Add heating setpoint with decoded zone name
                    decoded_zone = urllib.parse.unquote(zone)
                    zone_setpoints[zone].append({
                        "setpoint_type": "heating",
                        "actuator_key": f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                        "schedule_name": heating_schedule,
                        "control_type": "DualSetpoint"
                    })
                    
                    # Add cooling setpoint
                    zone_setpoints[zone].append({
                        "setpoint_type": "cooling",
                        "actuator_key": f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                        "schedule_name": cooling_schedule,
                        "control_type": "DualSetpoint"
                    })
            
            elif control_type == "ThermostatSetpoint:SingleHeating":
                # Get heating schedule name
                heating_query = f"""
                SELECT ?schedule
                WHERE {{
                    ?setpoint a ?type .
                    FILTER(?type = "ThermostatSetpoint:SingleHeating") .
                    FILTER(?setpoint = "{setpoint_name}"^^<http://www.w3.org/2001/XMLSchema#string>) .
                    ?setpoint ?schedule_prop ?schedule .
                    FILTER(?schedule_prop = ns:setpoint_temperature_schedule_name) .
                }}
                """
                
                for row in rdf_graph.query(heating_query, initNs={"ns": ns}):
                    schedule = str(row.schedule)
                    decoded_zone = urllib.parse.unquote(zone)
                    
                    zone_setpoints[zone].append({
                        "setpoint_type": "heating",
                        "actuator_key": f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleHeating"
                    })
            
            elif control_type == "ThermostatSetpoint:SingleCooling":
                # Get cooling schedule name
                cooling_query = f"""
                SELECT ?schedule
                WHERE {{
                    ?setpoint a ?type .
                    FILTER(?type = "ThermostatSetpoint:SingleCooling") .
                    FILTER(?setpoint = "{setpoint_name}"^^<http://www.w3.org/2001/XMLSchema#string>) .
                    ?setpoint ?schedule_prop ?schedule .
                    FILTER(?schedule_prop = ns:setpoint_temperature_schedule_name) .
                }}
                """
                
                for row in rdf_graph.query(cooling_query, initNs={"ns": ns}):
                    schedule = str(row.schedule)
                    decoded_zone = urllib.parse.unquote(zone)
                    
                    zone_setpoints[zone].append({
                        "setpoint_type": "cooling",
                        "actuator_key": f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleCooling"
                    })
            
            elif control_type == "ThermostatSetpoint:SingleHeatingOrCooling":
                # Get schedule name
                schedule_query = f"""
                SELECT ?schedule
                WHERE {{
                    ?setpoint a ?type .
                    FILTER(?type = "ThermostatSetpoint:SingleHeatingOrCooling") .
                    FILTER(?setpoint = "{setpoint_name}"^^<http://www.w3.org/2001/XMLSchema#string>) .
                    ?setpoint ?schedule_prop ?schedule .
                    FILTER(?schedule_prop = ns:setpoint_temperature_schedule_name) .
                }}
                """
                
                for row in rdf_graph.query(schedule_query, initNs={"ns": ns}):
                    schedule = str(row.schedule)
                    decoded_zone = urllib.parse.unquote(zone)
                    
                    # This type can switch between heating and cooling, so create both actuator keys
                    zone_setpoints[zone].append({
                        "setpoint_type": "heating",
                        "actuator_key": f"Zone Temperature Control,Temperature Heating Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleHeatingOrCooling"
                    })
                    
                    zone_setpoints[zone].append({
                        "setpoint_type": "cooling",
                        "actuator_key": f"Zone Temperature Control,Temperature Cooling Setpoint,{decoded_zone}",
                        "schedule_name": schedule,
                        "control_type": "SingleHeatingOrCooling"
                    })
    
    return zone_setpoints
