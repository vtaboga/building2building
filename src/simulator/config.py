"""To run an EnergyPlus simulation and wrap it into the neat

observation -> policy -> action -> environment -> observation

box commonly used in RL, it is necessary to have the following components:

1. A building file.
2. A weather file.
3. The set of EP variables which will be written to the observations.
4. The set of EP actuators which will be read from the action.

This module collects functions useful to creates those 4-tuples.

"""

import src.simulator.query_info as query_info
import rdflib
import typing
import src.simulator.simulation as simulation
from typing import Dict, List
from src.simulator.query_info import ns
import urllib.parse
import logging
from src.core.logging import setup_logger

# Setup config logger as a child of root
logger = setup_logger('config', add_handlers=False)

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
        # URL-decode the zone name
        decoded_zone = urllib.parse.unquote(z)
        heating[z] = simulation.VariableHole(
            "Zone Thermostat Heating Setpoint Temperature", decoded_zone
        )
        cooling[z] = simulation.VariableHole(
            "Zone Thermostat Cooling Setpoint Temperature", decoded_zone
        )


def auto_add_temperature(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
) -> None:
    """Add zone air temperatures to the observation template."""
    if "temperature" not in obs_template:
        obs_template["temperature"] = {}

    temps = obs_template["temperature"]
    for z in query_info.rdf_zones(rdf):
        # URL-decode the zone name
        decoded_zone = urllib.parse.unquote(z)
        temps[z] = simulation.VariableHole("ZONE AIR TEMPERATURE", decoded_zone)
        

def auto_add_energy(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
) -> None:
    """Add HVAC energy consumption meters to the observation template."""
    if "energy" not in obs_template:
        obs_template["energy"] = {}

    energy = obs_template["energy"]
    # Add whole building HVAC energy meters only
    energy["HVAC_electricity"] = simulation.MeterHole("Electricity:HVAC")
    energy["HVAC_natural_gas"] = simulation.MeterHole("NaturalGas:HVAC")


def auto_add_time(
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
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
    rdf: rdflib.Graph, obs_template: typing.Dict[str, typing.Any]
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