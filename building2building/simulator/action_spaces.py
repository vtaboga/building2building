import logging
import urllib.parse
from dataclasses import dataclass

import minergym.ontology as ontology
import numpy as np
from gymnasium.spaces import Box, Dict
from minergym.simulation import ActuatorHole
from rdflib.term import Node

from .transform_utils import (
    Transform,
    TransformConcat,
    TransformDictSpace,
    TransformList,
    TransformListToArray,
    TransformListToArrayShift,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HVACActuator:
    """Represents a controllable HVAC actuator in EnergyPlus."""
    name: str
    actuator_type: str  # Component type (e.g., "Ideal Loads Air System")
    control_type: str   # Control variable (e.g., "Air Mass Flow Rate")
    actuated_component: str  # Component name in the model
    min_value: float = 0.0
    max_value: float = float('inf')


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


ThermostatSetpoint = (
    DualSetpoint | SingleHeating | SingleCooling | SingleHeatingOrCooling
)


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
        control_type_str: str = control_type.toPython()
        setpoint: Node = setpoint_results[0].setpoint
        logger.debug(f"Control type: {control_type}")
        logger.debug(f"Setpoint name: {setpoint}")

        # Handle different thermostat setpoint types
        for zone in zones:
            zone_name = str(zone)

            if zone_name not in zone_setpoints:
                zone_setpoints[zone_name] = []

            if control_type_str == "ThermostatSetpoint:DualSetpoint":
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

            elif control_type_str == "ThermostatSetpoint:SingleHeating":
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

            elif control_type_str == "ThermostatSetpoint:SingleCooling":
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

            elif control_type_str == "ThermostatSetpoint:SingleHeatingOrCooling":
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
                raise Exception(f"unexpected control type: {control_type}")

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
                [ActuatorHole("Schedule:Compact", "Schedule Value", t.schedule)],
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


def many_thermostats_transform_dict(
    setpoints: list[ThermostatSetpoint],
) -> Transform[dict, Dict]:
    d: dict[str, Transform[list, Box]] = {}
    for sp in setpoints:
        name = sp.name
        d[name] = single_thermostat_transform(sp)
    return TransformDictSpace(d)


def get_hvac_actuators(ont: ontology.Ontology) -> dict[str, list[HVACActuator]]:
    """
    Extracts controllable HVAC actuators from an EnergyPlus building ontology.
    
    These actuators control the sensible heating/cooling output to zones, which is
    the actual heat transfer that maintains zone temperatures at the setpoints.
    
    The function identifies several types of HVAC equipment:
    1. Ideal Loads Air Systems - Direct control of sensible heating/cooling rates
    2. ZoneHVAC equipment (fan coils, baseboards, unit heaters, etc.)
    3. AirLoopHVAC equipment (central air systems with coils)
    4. Plant equipment (boilers, chillers) that serve zones
    
    Args:
        ont: EnergyPlus building ontology
        
    Returns:
        Dictionary mapping zone names to lists of HVACActuator objects
        
    Note:
        The actuators returned here operate at a lower level than thermostat 
        setpoints. While setpoints define target temperatures, these actuators 
        control the actual equipment that delivers heating/cooling to meet those 
        targets.
    """
    zone_actuators: dict[str, list[HVACActuator]] = {}
    
    # Initialize with all zones
    zones = [str(zone) for zone in ont.zones()]
    for zone in zones:
        zone_actuators[zone] = []
    
    logger.info("Searching for HVAC actuators in the building model...")
    
    # 1. Check for Ideal Loads Air Systems (ZoneHVAC:IdealLoadsAirSystem)
    # These provide direct control over sensible heating/cooling rates
    ideal_loads_query = """
    SELECT ?name ?zone
    WHERE {
        ?name a "ZoneHVAC:IdealLoadsAirSystem" .
        ?name idf:zone_name ?zone .
    }
    """
    
    logger.debug("Querying for Ideal Loads Air Systems...")
    for row in ont.rdf.query(ideal_loads_query):
        component_name = str(row.name)
        zone_name = str(row.zone)
        
        if zone_name in zone_actuators:
            # Ideal loads can control heating and cooling rates directly
            zone_actuators[zone_name].append(
                HVACActuator(
                    name=f"{component_name}_heating",
                    actuator_type="Ideal Loads Air System",
                    control_type="Air Mass Flow Rate",
                    actuated_component=component_name,
                    min_value=0.0,
                    max_value=100.0  # kg/s, should be determined from design
                )
            )
            logger.debug(f"Found Ideal Loads system in zone {zone_name}")
    
    # 2. Check for ZoneHVAC equipment types
    # Common zone equipment that can be controlled
    zone_hvac_types = [
        "ZoneHVAC:FourPipeFanCoil",
        "ZoneHVAC:PackagedTerminalAirConditioner", 
        "ZoneHVAC:PackagedTerminalHeatPump",
        "ZoneHVAC:WaterToAirHeatPump",
        "ZoneHVAC:Baseboard:Convective:Electric",
        "ZoneHVAC:Baseboard:Convective:Water",
        "ZoneHVAC:Baseboard:RadiantConvective:Electric",
        "ZoneHVAC:Baseboard:RadiantConvective:Water",
        "ZoneHVAC:UnitHeater",
        "ZoneHVAC:UnitVentilator",
    ]
    
    for hvac_type in zone_hvac_types:
        zone_hvac_query = f"""
        SELECT ?name ?zone
        WHERE {{
            ?name a "{hvac_type}" .
            ?name idf:availability_schedule_name ?schedule .
        }}
        """
        
        # Try to find zone association (property names vary by equipment type)
        zone_property_names = [
            "idf:zone_name",
            "idf:zone_supply_air_node_name", 
            "idf:air_inlet_node_name",
        ]
        
        for row in ont.rdf.query(zone_hvac_query):
            component_name = str(row.name)
            
            # Try to find which zone this equipment serves
            for prop in zone_property_names:
                zone_query = f"""
                SELECT ?zone
                WHERE {{
                    ?comp {prop} ?zone .
                }}
                """
                zone_results = list(ont.rdf.query(zone_query, 
                                                   initBindings={"comp": row.name}))
                if zone_results:
                    zone_name = str(zone_results[0].zone)
                    if zone_name in zone_actuators:
                        zone_actuators[zone_name].append(
                            HVACActuator(
                                name=component_name,
                                actuator_type=hvac_type,
                                control_type="Availability Status",
                                actuated_component=component_name,
                                min_value=0.0,  # Off
                                max_value=1.0,  # On
                            )
                        )
                        logger.debug(f"Found {hvac_type} in zone {zone_name}")
                    break
    
    # 3. Check for Coils (heating and cooling)
    # Coils are the primary components that add/remove heat
    coil_types = [
        ("Coil:Heating:Electric", "Electric Heating Coil Power"),
        ("Coil:Heating:Fuel", "Heating Coil Power"),
        ("Coil:Heating:Water", "Water Mass Flow Rate"),
        ("Coil:Heating:Steam", "Steam Mass Flow Rate"),
        ("Coil:Cooling:Water", "Water Mass Flow Rate"),
        ("Coil:Cooling:Water:DetailedGeometry", "Water Mass Flow Rate"),
        ("Coil:Cooling:DX:SingleSpeed", "Coil Speed Level"),
        ("Coil:Cooling:DX:TwoSpeed", "Coil Speed Level"),
        ("Coil:Cooling:DX:MultiSpeed", "Coil Speed Level"),
        ("Coil:Heating:DX:SingleSpeed", "Coil Speed Level"),
    ]
    
    for coil_type, control_type in coil_types:
        coil_query = f"""
        SELECT ?name
        WHERE {{
            ?name a "{coil_type}" .
        }}
        """
        
        for row in ont.rdf.query(coil_query):
            component_name = str(row.name)
            
            # Coils are typically part of air loops or zone equipment
            # We would need to trace connections to find which zones they serve
            # For now, we log them as available actuators
            logger.debug(f"Found {coil_type}: {component_name}")
            
            # Note: Without zone association, we can't add these to zone_actuators
            # A more sophisticated implementation would trace the air loop topology
    
    # 4. Check for AirLoopHVAC systems
    # These are central systems that serve multiple zones
    airloop_query = """
    SELECT ?name
    WHERE {
        ?name a "AirLoopHVAC" .
    }
    """
    
    for row in ont.rdf.query(airloop_query):
        airloop_name = str(row.name)
        logger.debug(f"Found AirLoopHVAC: {airloop_name}")
        
        # To properly handle air loops, we would need to:
        # 1. Find the supply side equipment (fans, coils)
        # 2. Trace connections to zone terminals
        # 3. Associate actuators with served zones
        # This requires more complex topology analysis
    
    # Log summary
    total_actuators = sum(len(acts) for acts in zone_actuators.values())
    zones_with_actuators = [z for z, acts in zone_actuators.items() if acts]
    zones_without_actuators = [z for z, acts in zone_actuators.items() if not acts]
    
    logger.info(f"Found {total_actuators} HVAC actuators across {len(zones_with_actuators)} zones")
    if zones_without_actuators:
        logger.warning(f"No HVAC actuators found for zones: {zones_without_actuators}")
    
    return zone_actuators
