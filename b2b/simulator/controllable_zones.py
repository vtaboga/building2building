"""Functions to extract which zones are controlled by HVAC actuators.

These functions mirror the make_*_controllable functions in actuators.py,
using similar SPARQL queries to determine which zones are served by
controllable HVAC equipment.

In the future, this information should be integrated directly into the
ActuatorDescription objects and produced in the associated make_*_controllable
function. This will make the relation between an actuator and the observation
(per-zone) associated to it easier to access, which is important for certain
learning algorithms which rely on an accuration representation of the
morphology.

"""

from typing import Any

from minergym.ontology import Ontology


def get_unitary_hvac_controllable_zones(obj: dict[str, Any]) -> list[str]:
    """Find all zones controlled by AirLoopHVAC:UnitarySystem equipment.

    This uses the controlling_zone_or_thermostat_location field from each
    unitary system to determine which zone it controls.

    Args:
        obj: epJSON dictionary (should be the output of make_controllable)

    Returns:
        List of unique zone names controlled by unitary HVAC systems
    """
    ont = Ontology.from_object(obj)

    # Query mirrors make_unitary_hvac_controllable() but extracts zone info
    zones_query = """# -*- mode: sparql -*-
SELECT ?loop ?zone
WHERE {
  ?loop a "AirLoopHVAC:UnitarySystem" .
  ?loop idf:controlling_zone_or_thermostat_location ?zone .
}
"""

    zones = set()
    for loop, zone in ont.rdf.query(zones_query):
        zone_name = str(zone)
        zones.add(zone_name)

    return sorted(zones)


def get_baseboard_controllable_zones(obj: dict[str, Any]) -> list[str]:
    """Find all zones served by ZoneHVAC:Baseboard:Convective:Electric equipment.

    Traces from baseboard equipment through ZoneHVAC:EquipmentList to
    ZoneHVAC:EquipmentConnections to find which zones contain each baseboard.

    Args:
        obj: epJSON dictionary (should be the output of make_controllable)

    Returns:
        List of unique zone names containing baseboard heaters
    """
    ont = Ontology.from_object(obj)

    # Query traces baseboard -> equipment list -> equipment connections -> zone
    zones_query = """# -*- mode: sparql -*-
SELECT ?zone ?baseboard
WHERE {
  ?baseboard a "ZoneHVAC:Baseboard:Convective:Electric" .

  # Find which equipment list contains this baseboard
  ?list a "ZoneHVAC:EquipmentList" .
  ?list idf:equipment ?eq_entry .
  ?eq_entry idf:zone_equipment_name ?baseboard .

  # Find which zone uses this equipment list
  ?connections a "ZoneHVAC:EquipmentConnections" .
  ?connections idf:zone_conditioning_equipment_list_name ?list .
  ?connections idf:zone_name ?zone .
}
"""

    zones = set()
    for zone, baseboard in ont.rdf.query(zones_query):
        zone_name = str(zone)
        zones.add(zone_name)

    return sorted(zones)


def get_fanonoff_controllable_zones(obj: dict[str, Any]) -> list[str]:
    """Find zones served by systems using Fan:OnOff equipment.

    Fan:OnOff objects are typically components of larger systems
    (like AirLoopHVAC:UnitarySystem). This function traces fans to their
    parent systems and returns the zones controlled by those systems.

    Args:
        obj: epJSON dictionary (should be the output of make_controllable)

    Returns:
        List of unique zone names served by systems with Fan:OnOff components
    """
    ont = Ontology.from_object(obj)

    # Strategy: Find UnitarySystems that reference Fan:OnOff objects,
    # then get their controlling zones
    zones_query = """# -*- mode: sparql -*-
SELECT DISTINCT ?zone ?fan
WHERE {
  ?fan a "Fan:OnOff" .

  # Find unitary systems that use this fan
  ?system a "AirLoopHVAC:UnitarySystem" .
  ?system idf:supply_fan_name ?fan .

  # Get the controlling zone
  ?system idf:controlling_zone_or_thermostat_location ?zone .
}
"""

    zones = set()
    for zone, fan in ont.rdf.query(zones_query):
        zone_name = str(zone)
        zones.add(zone_name)

    return sorted(zones)


def get_all_controllable_zones(obj: dict[str, Any]) -> list[str]:
    """Get all zones controlled by any controllable HVAC equipment.

    This aggregates zones from all get_*_controllable_zones functions and
    returns a deduplicated, sorted list.

    Args:
        obj: epJSON dictionary (should be the output of make_controllable)

    Returns:
        Sorted list of unique zone names controlled by any HVAC actuators
    """
    zones = set()

    # Collect zones from all equipment types
    zones.update(get_unitary_hvac_controllable_zones(obj))
    zones.update(get_baseboard_controllable_zones(obj))
    zones.update(get_fanonoff_controllable_zones(obj))

    return sorted(zones)
