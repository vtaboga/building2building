import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from cattrs import structure, unstructure
from minergym.ontology import Ontology

from b2b.store import (
    OUTPUT,
    Expression,
    Realizable,
    derivation,
    expression,
)
from b2b.types import ActuatorDescription


@dataclass
class Gensym:
    i: int = 0

    def __call__(self) -> int:
        out = self.i
        self.i += 1
        return out


# global counter for schedule type limits and constants
# Use this to ensure that the names are unique across all components.
_DEFAULT_GENSYM = Gensym()


def create_onoff_availability_stl(
    obj: dict[str, Any], *, name: str = "OnOff", gensym: Gensym | None = None
) -> str:
    """Create a binary ScheduleTypeLimits entity and return its name."""
    gensym = _DEFAULT_GENSYM if gensym is None else gensym
    schedule_type_limits = obj.setdefault("ScheduleTypeLimits", {})
    name = f"B2B {name} ({gensym()})"
    schedule_type_limits[name] = {
        "lower_limit_value": 0,
        "upper_limit_value": 1,
        "numeric_type": "Discrete",
        "unit_type": "Availability",
    }
    return name


temp_stl_lower_bound = 5.0
temp_stl_upper_bound = 50.0


def create_temp_stl(
    obj: dict[str, Any],
    *,
    name: str = "Temperature",
    gensym: Gensym | None = None,
    lower_limit_value: float = temp_stl_lower_bound,
    upper_limit_value: float = temp_stl_upper_bound,
) -> str:
    """Create a continuous ScheduleTypeLimits for temperatures and return its
    name.

    """
    gensym = _DEFAULT_GENSYM if gensym is None else gensym
    schedule_type_limits = obj.setdefault("ScheduleTypeLimits", {})

    name = f"B2B {name} ({gensym()})"
    schedule_type_limits[name] = {
        "lower_limit_value": float(lower_limit_value),
        "upper_limit_value": float(upper_limit_value),
        "numeric_type": "Continuous",
        "unit_type": "Temperature",
    }

    return name


def create_schedule_constant(
    obj: dict[str, Any],
    stl_name: str,
    hourly_value: float,
    *,
    name: str = "constant schedule",
    gensym: Gensym | None = None,
) -> str:
    """Create a constant schedule with the given type and the given constant
    value and return its name.

    """
    gensym = _DEFAULT_GENSYM if gensym is None else gensym
    schedule_constants = obj.setdefault("Schedule:Constant", {})
    name = f"B2B {name} ({gensym()})"
    schedule_constants[name] = {
        "hourly_value": hourly_value,
        "schedule_type_limits_name": stl_name,
    }

    return name


def make_unitary_hvac_controllable(
    obj: dict[str, Any],
    only_outlet_nodes: bool = False,
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all unitary air loops and expose the relevant
    node setpoints as schedules that can be controlled by minergym.

    This is done in many steps:

    1. We create the relevant schedule type descriptors (ScheduleTypeLimits)
       that will be used by all generated schedules. Those consist of

       1. an OnOff type which will be used by the system fan's schedule
          ("supply_air_fan_operating_mode_schedule_name").

       2. a Temperature type which will be used by all schedules we use for
          controlling temperature.

    2. We query the ontology and look for unitary air loops
       ("AirLoopHVAC:UnitarySystem" and "AirLoopHVAC:UnitaryHeatPump:AirToAir").
       For each of those, we do the following:

       1. We set the control_type to SetPoint

       2. We create a schedule to make the fan mode controllable.

       3. For each of the nodes associated to system (outlet_node,
          cooling_coil_node, heating_coil_node, supplemental_coil_node), we set
          up a scheduler and a setpoint manager that makes that specific node
          controllable.

    """

    obj = deepcopy(obj)

    new_actuators = []

    gensym = Gensym() if gensym is None else gensym
    ont = Ontology.from_object(obj)

    setpoint_managers = obj.setdefault("SetpointManager:Scheduled", {})

    # We define the schedule type descriptors (onoff and temperature)

    onoff_stl_name = create_onoff_availability_stl(
        obj, name="unitaryhvac fan availibiliby stl", gensym=gensym
    )
    temp_stl_name = create_temp_stl(
        obj, name="unitaryhvac temperature setpoints stl", gensym=gensym
    )

    # Note: I wrapped each coil section in OPTIONAL blocks because not all
    # unitary systems have all three coil types (e.g., cooling-only systems
    # won't have heating coils).
    #
    # TODO: actually handle cases where some of these are None.
    all_loops_query = """# -*- mode: sparql-*-
SELECT ?loop ?loop_type ?outlet_node ?cooling_coil ?cooling_coil_node ?heating_coil ?heating_coil_node ?supplemental_coil ?supplemental_coil_node
WHERE {
  VALUES ?loop_type { "AirLoopHVAC:UnitarySystem" "AirLoopHVAC:UnitaryHeatPump:AirToAir" } .
  ?loop a ?loop_type .
  ?loop idf:air_outlet_node_name ?outlet_node .

  # Cooling coil outlet
  OPTIONAL {
    ?loop idf:cooling_coil_name ?cooling_coil .
    ?loop idf:cooling_coil_object_type ?cooling_coil_type .
    ?cooling_coil a ?cooling_coil_type .

    # Try both possible outlet field names
    { ?cooling_coil idf:air_outlet_node_name ?cooling_coil_node }
    UNION
    { ?cooling_coil idf:outlet_node_name ?cooling_coil_node }
  }

  # Heating coil outlet
  OPTIONAL {
    ?loop idf:heating_coil_name ?heating_coil .
    ?loop idf:heating_coil_object_type ?heating_coil_type .
    ?heating_coil a ?heating_coil_type .

    { ?heating_coil idf:air_outlet_node_name ?heating_coil_node }
    UNION
    { ?heating_coil idf:outlet_node_name ?heating_coil_node }
  }

  # Supplemental heating coil outlet
  OPTIONAL {
    ?loop idf:supplemental_heating_coil_name ?supplemental_coil .
    ?loop idf:supplemental_heating_coil_object_type ?supplemental_coil_type .
    ?supplemental_coil a ?supplemental_coil_type .

    { ?supplemental_coil idf:air_outlet_node_name ?supplemental_coil_node }
    UNION
    { ?supplemental_coil idf:outlet_node_name ?supplemental_coil_node }
  }
}


"""
    # loop, outlet, cooling_coil, cooling_coil_node
    for (
        loop,
        loop_type,
        outlet_node,
        cooling_coil,
        cooling_coil_node,
        heating_coil,
        heating_coil_node,
        supplemental_coil,
        supplemental_coil_node,
    ) in ont.rdf.query(all_loops_query):
        loop_type_name = str(loop_type)
        unitary_system = obj[loop_type_name][str(loop)]

        unitary_system["control_type"] = "SetPoint"

        # Set up the fan mode. It should be always on
        fan_mode_schedule_name = create_schedule_constant(
            obj,
            onoff_stl_name,
            1,
            name="unitaryhvac fan mode schedule",
            gensym=gensym,
        )

        unitary_system["supply_air_fan_operating_mode_schedule_name"] = (
            fan_mode_schedule_name
        )

        # UnitaryHeatPump:AirToAir uses "supply_air_fan_name", while UnitarySystem
        # uses "supply_fan_name".
        if loop_type_name == "AirLoopHVAC:UnitaryHeatPump:AirToAir":
            supply_fan_name = unitary_system.get("supply_air_fan_name")
        else:
            supply_fan_name = unitary_system.get("supply_fan_name")

        if not isinstance(supply_fan_name, str) or not supply_fan_name.strip():
            # Keep going: setpoints can still be created even if fan name is missing.
            supply_fan_name = None

        # The fan air mass flow rate isn't acuated through a schedule, but
        # directly through an EnergyManagementSystem:Actuator.

        if supply_fan_name is not None:
            fan_air_mass_flow_rate = ActuatorDescription(
                "Fan",
                "Fan Air Mass Flow Rate",
                supply_fan_name,
                "[kg/s]",
                0,
                100,
            )
            new_actuators.append(fan_air_mass_flow_rate)

        # Some buildings legitimately have missing coil outlet nodes or explicit
        # "NONE" placeholders in node fields. Never create setpoint managers for
        # such nodes, otherwise EnergyPlus errors out with:
        #   Node Connection Error, Node="NONE", Setpoint node did not find a matching node...
        if only_outlet_nodes:
            raw_nodes = [("outlet", outlet_node)]
        else:
            raw_nodes = [
                ("outlet", outlet_node),
                ("cooling", cooling_coil_node),
                ("heating", heating_coil_node),
                ("reheat", supplemental_coil_node),
            ]

        # Keep track of where each setpoint node came from (outlet/cooling/heating/reheat)
        # for interpretability, while still deduplicating identical node names.
        node_to_roles: dict[str, set[str]] = {}
        for role, node in raw_nodes:
            if node is None:
                continue
            s = str(node).strip()
            if not s or s.upper() == "NONE":
                continue
            node_to_roles.setdefault(s, set()).add(role)

        if not node_to_roles:
            logger.warning(
                "Unitary system %s has no valid setpoint nodes (skipping SPM creation).",
                str(loop),
            )
            continue

        for node_name in sorted(node_to_roles):
            roles = sorted(node_to_roles[node_name])
            roles_str = "+".join(roles)

            sched_constant_name = create_schedule_constant(
                obj,
                temp_stl_name,
                22,
                name=f"unitaryhvac {roles_str} temp setpoint schedule",
                gensym=gensym,
            )

            setpoint_manager_name = (
                f"B2B {roles_str} Node TEMP SPM for {node_name} ({gensym()})"
            )

            setpoint_managers[setpoint_manager_name] = {
                "control_variable": "Temperature",
                "schedule_name": sched_constant_name,
                "setpoint_node_or_nodelist_name": node_name,
            }

            new_actuators.append(
                ActuatorDescription(
                    component_type="Schedule:Constant",
                    control_type="Schedule Value",
                    component_name=sched_constant_name,
                    units="Temperature",
                    lower_bound=temp_stl_lower_bound,
                    upper_bound=temp_stl_upper_bound,
                )
            )

    return obj, new_actuators


def make_baseboard_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all baseboards and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(
        obj, name="baseboard availibility", gensym=gensym
    )

    all_baseboards_query = """# -*- mode: sparql -*-
SELECT ?baseboard WHERE {
  ?baseboard a "ZoneHVAC:Baseboard:Convective:Electric" .
}"""

    new_actuators = []

    for (baseboard,) in ont.rdf.query(all_baseboards_query):
        baseboard_name = baseboard.toPython()
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name="controllable schedule for baseboard",
            gensym=gensym,
        )

        obj["ZoneHVAC:Baseboard:Convective:Electric"][baseboard_name][
            "availability_schedule_name"
        ] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Availability",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators


def make_fanonoff_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all Fan:OnOff objects and expose the availability schedule as a
    schedule that can be controlled."""

    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(
        obj, name="fanonoff availability", gensym=gensym
    )

    all_fans_query = """# -*- mode: sparql -*-
SELECT ?fan WHERE {
  ?fan a "Fan:OnOff" .
}"""

    new_actuators = []

    for (fan,) in ont.rdf.query(all_fans_query):
        fan_name = fan.toPython()
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name="controllable schedule for fanonoff",
            gensym=gensym,
        )

        obj["Fan:OnOff"][fan_name]["availability_schedule_name"] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Availability",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators

'''def make_waterheater_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
     """Find all waterheaters and for each of those, expose the availibility
        schedue as a schedule that can be controlled."""
     
     obj = deepcopy(obj)

     ont = Ontology.from_object(obj)

     binary_stl = create_onoff_availability_stl(obj, name="baseboard availibility")
'''

def make_waterheater_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all waterheaters and for each of those, expose the availibility
        schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    ont = Ontology.from_object(obj)
    
    # Water heater setpoints are typically higher than space HVAC setpoints.
    # Align ScheduleTypeLimits bounds with the actuator bounds to avoid E+ fatal
    # errors during ProcessScheduleInput.
    temp_stl_name = create_temp_stl(
        obj,
        name="water heater temperature stl",
        gensym=gensym,
        lower_limit_value=40.0,
        upper_limit_value=70.0,
    )

    # SPARQL query for Water Heaters 
    all_waterheaters_query = """# -*- mode: sparql -*-
    SELECT ?wh WHERE {
      ?wh a "WaterHeater:Mixed" .
    }"""

    new_actuators = []

    for (wh_id,) in ont.rdf.query(all_waterheaters_query):
        wh_name = str(wh_id)
        wh_entry = obj["WaterHeater:Mixed"][wh_name]

        # Create a new controllable schedule for the setpoint 
        sched_name = create_schedule_constant(
            obj,
            temp_stl_name,
            60.0,
            name=f"controllable setpoint for {wh_name}",
            gensym=gensym,
        )
        
        # Override the original schedule 
        wh_entry["setpoint_temperature_schedule_name"] = sched_name

        new_actuators.append(
            ActuatorDescription(
                component_type="WaterHeater",
                control_type="Setpoint Temperature",
                component_name=sched_name,
                units="Temperature",
                lower_bound=10.0,
                upper_bound=80.0,
            )
        )

    return obj, new_actuators

def make_pump_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all Pump:ConstantSpeed and for each of those, expose the availibility
        schedue as a schedule that can be controlled."""
    
    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    new_actuators = []
    
    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(
        obj, name="pump availability stl", gensym=gensym
    )

    # SPARQL query for Constant Speed Pumps
    pump_query = """# -*- mode: sparql -*-
    SELECT ?pump WHERE {
      ?pump a "Pump:ConstantSpeed" .
    }"""

    for (pump_id,) in ont.rdf.query(pump_query):
        pump_name = str(pump_id)
        # In EnergyPlus pumps are often controlled via availability schedules
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name=f"controllable schedule for pump {pump_name}",
            gensym=gensym,
        )

        # Set the pump to use this new schedule (Adding field if not present)
        obj["Pump:ConstantSpeed"][pump_name]["pump_scheduling_control_scheme"] = "Schedule"
        obj["Pump:ConstantSpeed"][pump_name]["availability_schedule_name"] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Availability",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators

def make_airterminal_controllable(
    obj: dict[str, Any],    
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all ConstantVolume:NoReheat Air Terminals and for each of those, expose the availibility
        schedue as a schedule that can be controlled."""
    
    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    new_actuators = []
    
    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(obj, name="terminal availability stl", gensym=gensym)

    terminal_query = """# -*- mode: sparql -*-
    SELECT ?terminal WHERE {
      ?terminal a "AirTerminal:SingleDuct:ConstantVolume:NoReheat" .
    }"""

    for (term_id,) in ont.rdf.query(terminal_query):
        term_name = str(term_id)
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name=f"controllable schedule for terminal {term_name}",
            gensym=gensym,
        )

        obj["AirTerminal:SingleDuct:ConstantVolume:NoReheat"][term_name][
            "availability_schedule_name"
        ] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Availability",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators

def make_controller_outdoorair_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all Controller:OutdoorAir and for each of those, expose the availibility
        schedue as a schedule that can be controlled."""
     
    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym 

    new_actuators = []
    
    ont = Ontology.from_object(obj)

    fraction_stl = obj.get("ScheduleTypeLimits", {}).get("Fraction", None)
    if not fraction_stl:
        # Create a fraction STL if it doesn't exist for the controller
        schedule_type_limits = obj.setdefault("ScheduleTypeLimits", {})
        fraction_stl = f"B2B Fraction STL ({gensym()})"
        schedule_type_limits[fraction_stl] = {
            "lower_limit_value": 0,
            "upper_limit_value": 1,
            "numeric_type": "Continuous",
        }

    oa_controller_query = """# -*- mode: sparql -*-
    SELECT ?controller WHERE {
      ?controller a "Controller:OutdoorAir" .
    }"""

    for (ctrl_id,) in ont.rdf.query(oa_controller_query):
        ctrl_name = str(ctrl_id)
        new_schedule_name = create_schedule_constant(
            obj,
            str(fraction_stl),
            1,
            name=f"controllable OA fraction for {ctrl_name}",
            gensym=gensym,
        )

        obj["Controller:OutdoorAir"][ctrl_name][
            "minimum_outdoor_air_schedule_name"
        ] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Fraction",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators

'''def make_waterheater_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
     """Find all waterheaters and for each of those, expose the availibility
        schedue as a schedule that can be controlled."""
     
     obj = deepcopy(obj)

     ont = Ontology.from_object(obj)

     binary_stl = create_onoff_availability_stl(obj, name="baseboard availibility")
'''


def make_waterheater_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all waterheaters and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    ont = Ontology.from_object(obj)

    # Water heater setpoints are typically higher than space HVAC setpoints.
    # Align ScheduleTypeLimits bounds with the actuator bounds to avoid E+ fatal
    # errors during ProcessScheduleInput.
    temp_stl_name = create_temp_stl(
        obj,
        name="water heater temperature stl",
        gensym=gensym,
        lower_limit_value=40.0,
        upper_limit_value=70.0,
    )

    # SPARQL query for Water Heaters
    all_waterheaters_query = """# -*- mode: sparql -*-
    SELECT ?wh WHERE {
      ?wh a "WaterHeater:Mixed" .
    }"""

    new_actuators = []

    for (wh_id,) in ont.rdf.query(all_waterheaters_query):
        wh_name = str(wh_id)
        wh_entry = obj["WaterHeater:Mixed"][wh_name]

        # Create a new controllable schedule for the setpoint
        sched_name = create_schedule_constant(
            obj,
            temp_stl_name,
            60.0,
            name=f"controllable setpoint for {wh_name}",
            gensym=gensym,
        )

        # Override the original schedule
        wh_entry["setpoint_temperature_schedule_name"] = sched_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=sched_name,
                units="Temperature",
                lower_bound=40.0,
                upper_bound=70.0,
            )
        )

    return obj, new_actuators


def make_pump_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all Pump:ConstantSpeed and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    new_actuators = []

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(
        obj, name="pump availability stl", gensym=gensym
    )

    # SPARQL query for Constant Speed Pumps
    pump_query = """# -*- mode: sparql -*-
    SELECT ?pump WHERE {
      ?pump a "Pump:ConstantSpeed" .
    }"""

    for (pump_id,) in ont.rdf.query(pump_query):
        pump_name = str(pump_id)
        # In EnergyPlus pumps are often controlled via availability schedules
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name=f"controllable schedule for pump {pump_name}",
            gensym=gensym,
        )

        # Set the pump to use this new schedule (Adding field if not present)
        obj["Pump:ConstantSpeed"][pump_name][
            "pump_scheduling_control_scheme"
        ] = "Schedule"
        obj["Pump:ConstantSpeed"][pump_name][
            "availability_schedule_name"
        ] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Availability",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators


def make_airterminal_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all ConstantVolume:NoReheat Air Terminals and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    new_actuators = []

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(
        obj, name="terminal availability stl", gensym=gensym
    )

    terminal_query = """# -*- mode: sparql -*-
    SELECT ?terminal WHERE {
      ?terminal a "AirTerminal:SingleDuct:ConstantVolume:NoReheat" .
    }"""

    for (term_id,) in ont.rdf.query(terminal_query):
        term_name = str(term_id)
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name=f"controllable schedule for terminal {term_name}",
            gensym=gensym,
        )

        obj["AirTerminal:SingleDuct:ConstantVolume:NoReheat"][term_name][
            "availability_schedule_name"
        ] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Availability",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators


def make_controller_outdoorair_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all Controller:OutdoorAir and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)
    gensym = Gensym() if gensym is None else gensym

    new_actuators = []

    ont = Ontology.from_object(obj)

    fraction_stl = obj.get("ScheduleTypeLimits", {}).get("Fraction", None)
    if not fraction_stl:
        # Create a fraction STL if it doesn't exist for the controller
        schedule_type_limits = obj.setdefault("ScheduleTypeLimits", {})
        fraction_stl = f"B2B Fraction STL ({gensym()})"
        schedule_type_limits[fraction_stl] = {
            "lower_limit_value": 0,
            "upper_limit_value": 1,
            "numeric_type": "Continuous",
        }

    oa_controller_query = """# -*- mode: sparql -*-
    SELECT ?controller WHERE {
      ?controller a "Controller:OutdoorAir" .
    }"""

    for (ctrl_id,) in ont.rdf.query(oa_controller_query):
        ctrl_name = str(ctrl_id)
        new_schedule_name = create_schedule_constant(
            obj,
            str(fraction_stl),
            1,
            name=f"controllable OA fraction for {ctrl_name}",
            gensym=gensym,
        )

        obj["Controller:OutdoorAir"][ctrl_name][
            "minimum_outdoor_air_schedule_name"
        ] = new_schedule_name

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=new_schedule_name,
                units="Fraction",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        )

    return obj, new_actuators


def make_controllable(
    input_epjson: Realizable,
    *,
    controls: (
        Sequence[
            Literal[
                "unitary_hvac",
                "baseboard",
                "fanonoff",
                "waterheater",
                "pump",
                "airterminal",
                "controller_outdoorair",
            ]
        ]
        | None
    ) = None,
) -> Expression[tuple[Path, list[ActuatorDescription]]]:
    # By default, we enable all controls.
    # Keep the controls argument for backwards compatibility until code is stable
    selected_controls = (
        list(controls)
        if controls is not None
        else [
            "unitary_hvac",
            "baseboard",
            "fanonoff",
            "waterheater",
            "pump",
            "airterminal",
            "controller_outdoorair",
        ]
    )

    @derivation("controllable-building")
    def make_controllable_builder(input: Path, controls: list[str]):
        real_out = OUTPUT.get()
        with open(input, "rb") as f:
            json_obj = json.load(f)

        gensym = Gensym()
        # IMPORTANT: `controls` must be a derivation argument (not a closure),
        # so it is included in the derivation hash and caching is correct.
        selected = set(controls)
        all_actuators: list[ActuatorDescription] = []

        def _enabled(name: str) -> bool:
            return name in selected

        if _enabled("unitary_hvac"):
            json_obj, hvac_actuators = make_unitary_hvac_controllable(
                json_obj, only_outlet_nodes=True, gensym=gensym
            )
            all_actuators.extend(hvac_actuators)

        if _enabled("baseboard"):
            json_obj, baseboard_actuators = make_baseboard_controllable(
                json_obj, gensym=gensym
            )
            all_actuators.extend(baseboard_actuators)

        if _enabled("fanonoff"):
            json_obj, fanonoff_actuators = make_fanonoff_controllable(
                json_obj, gensym=gensym
            )
            all_actuators.extend(fanonoff_actuators)

        if _enabled("waterheater"):
            json_obj, waterheater_actuators = make_waterheater_controllable(
                json_obj, gensym=gensym
            )
            all_actuators.extend(waterheater_actuators)

        if _enabled("pump"):
            json_obj, pump_actuators = make_pump_controllable(json_obj, gensym=gensym)
            all_actuators.extend(pump_actuators)

        if _enabled("airterminal"):
            json_obj, airterminal_actuators = make_airterminal_controllable(
                json_obj, gensym=gensym
            )
            all_actuators.extend(airterminal_actuators)

        if _enabled("controller_outdoorair"):
            json_obj, controller_outdoorair_actuators = (
                make_controller_outdoorair_controllable(json_obj, gensym=gensym)
            )
            all_actuators.extend(controller_outdoorair_actuators)

        tmp_out = Path(tempfile.mkdtemp())

        # out.mkdir()

        json.dump(json_obj, open(tmp_out / "building.epjson", "w"), indent=4)
        json.dump(
            unstructure(all_actuators),
            open(tmp_out / "actuators.json", mode="w"),
            indent=4,
        )

        shutil.move(tmp_out, real_out)

    @expression()
    def parse_expr(folder: Path) -> tuple[Path, list[ActuatorDescription]]:
        with open(folder / "actuators.json", mode="r") as f:
            actuators_json = json.load(f)

        return folder / "building.epjson", structure(
            actuators_json, list[ActuatorDescription]
        )

    return parse_expr(make_controllable_builder(input_epjson, selected_controls))
