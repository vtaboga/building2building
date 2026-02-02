import json
import logging
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cattrs import structure, unstructure
from minergym.ontology import Ontology

from building2building.store import (
    OUTPUT,
    Expression,
    Realizable,
    derivation,
    expression,
)
from building2building.types import ActuatorDescription

logger = logging.getLogger(__name__)


@dataclass
class Gensym:
    i: int = 0

    def __call__(self) -> int:
        out = self.i
        self.i += 1
        return out


gensym = Gensym()


def create_onoff_availability_stl(obj: dict[str, Any], *, name="OnOff") -> str:
    """Create a binary ScheduleTypeLimits entity and return its name."""
    schedule_type_limits = obj.setdefault("ScheduleTypeLimits", {})
    name = f"B2B {name} ({gensym()})"
    schedule_type_limits[name] = {
        "lower_limit_value": 0,
        "upper_limit_value": 1,
        "numeric_type": "Discrete",
        "unit_type": "Availability",
    }
    return name


temp_stl_lower_bound = -100.0
temp_stl_upper_bound = 200.0


def create_temp_stl(obj: dict[str, Any], *, name="Temperature") -> str:
    """Create a continuous ScheduleTypeLimits for temperatures and return its
    name.

    """
    schedule_type_limits = obj.setdefault("ScheduleTypeLimits", {})

    name = f"B2B {name} ({gensym()})"
    schedule_type_limits[name] = {
        "lower_limit_value": -100,
        "upper_limit_value": 200,
        "numeric_type": "Continuous",
        "unit_type": "Temperature",
    }

    return name


def create_schedule_constant(
    obj: dict[str, Any], stl_name: str, hourly_value: int, *, name="constant schedule"
) -> str:
    """Create a constant schedule with the given type and the given constant
    value and return its name.

    """
    schedule_constants = obj.setdefault("Schedule:Constant", {})
    name = f"B2B {name} ({gensym()})"
    schedule_constants[name] = {
        "hourly_value": hourly_value,
        "schedule_type_limits_name": stl_name,
    }

    return name


def make_unitary_hvac_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all "AirLoopHVAC:UnitarySystem" and expose the relevant node
    setpoints as schedules that can be controlled by minergym.

    This is done in many steps:

    1. We create the relevant schedule type descriptors (ScheduleTypeLimits)
       that will be used by all generated schedules. Those consist of

       1. an OnOff type which will be used by the system fan's schedule
          ("supply_air_fan_operating_mode_schedule_name").

       2. a Temperature type which will be used by all schedules we use for
          controlling temperature.

    2. We query the ontology and look for all "AirLoopHVAC:UnitarySystem". For
       each of those, we do the following:

       1. We set the control_type to SetPoint

       2. We create a schedule to make the fan mode controllable.

       3. For each of the nodes associated to system (outlet_node,
          cooling_coil_node, heating_coil_node, supplemental_coil_node), we set
          up a scheduler and a setpoint manager that makes that specific node
          controllable.

    """

    obj = deepcopy(obj)

    new_actuators = []

    gensym = Gensym()
    ont = Ontology.from_object(obj)

    setpoint_managers = obj.setdefault("SetpointManager:Scheduled", {})

    # We define the schedule type descriptors (onoff and temperature)

    onoff_stl_name = create_onoff_availability_stl(
        obj, name="unitaryhvac fan availibiliby stl"
    )
    temp_stl_name = create_temp_stl(obj, name="unitaryhvac temperature setpoints stl")

    # Note: I wrapped each coil section in OPTIONAL blocks because not all
    # unitary systems have all three coil types (e.g., cooling-only systems
    # won't have heating coils).
    #
    # TODO: actually handle cases where some of these are None.
    all_loops_query = """# -*- mode: sparql-*-
SELECT ?loop ?outlet_node ?cooling_coil ?cooling_coil_node ?heating_coil ?heating_coil_node ?supplemental_coil ?supplemental_coil_node
WHERE {
  ?loop a "AirLoopHVAC:UnitarySystem" .
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
        outlet_node,
        cooling_coil,
        cooling_coil_node,
        heating_coil,
        heating_coil_node,
        supplemental_coil,
        supplemental_coil_node,
    ) in ont.rdf.query(all_loops_query):
        unitary_system = obj["AirLoopHVAC:UnitarySystem"][str(loop)]

        unitary_system["control_type"] = "SetPoint"

        # Set up the fan mode. It should be always on
        fan_mode_schedule_name = create_schedule_constant(
            obj,
            onoff_stl_name,
            1,
            name="unitaryhvac fan mode schedule",
        )

        unitary_system["supply_air_fan_operating_mode_schedule_name"] = (
            fan_mode_schedule_name
        )

        supply_fan_name = unitary_system["supply_fan_name"]

        # The fan air mass flow rate isn't acuated through a schedule, but
        # directly through an EnergyManagementSystem:Actuator.

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
        raw_nodes = [
            outlet_node,
            cooling_coil_node,
            heating_coil_node,
            supplemental_coil_node,
        ]
        node_names: set[str] = set()
        for node in raw_nodes:
            if node is None:
                continue
            s = str(node).strip()
            if not s or s.upper() == "NONE":
                continue
            node_names.add(s)

        if not node_names:
            logger.warning(
                "Unitary system %s has no valid setpoint nodes (skipping SPM creation).",
                str(loop),
            )
            continue

        for node_name in node_names:

            sched_constant_name = create_schedule_constant(
                obj,
                temp_stl_name,
                22,
                name="unitaryhvac schedule for node",
            )

            setpoint_manager_name = f"B2B Node TEMP SPM for {node_name} ({gensym()}) "

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
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all baseboards and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(obj, name="baseboard availibility")

    all_baseboards_query = """# -*- mode: sparql -*-
SELECT ?baseboard WHERE {
  ?baseboard a "ZoneHVAC:Baseboard:Convective:Electric" .
}"""

    new_actuators = []

    for (baseboard,) in ont.rdf.query(all_baseboards_query):
        baseboard_name = baseboard.toPython()
        new_schedule_name = create_schedule_constant(
            obj, binary_stl, 1, name="controllable schedule for baseboard"
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
) -> tuple[dict[str, Any], list[ActuatorDescription]]:
    """Find all Fan:OnOff objects and expose the availability schedule as a
    schedule that can be controlled."""

    obj = deepcopy(obj)

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(obj, name="fanonoff availability")

    all_fans_query = """# -*- mode: sparql -*-
SELECT ?fan WHERE {
  ?fan a "Fan:OnOff" .
}"""

    new_actuators = []

    for (fan,) in ont.rdf.query(all_fans_query):
        fan_name = fan.toPython()
        new_schedule_name = create_schedule_constant(
            obj, binary_stl, 1, name="controllable schedule for fanonoff"
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


def make_controllable(
    input_epjson: Realizable,
) -> Expression[tuple[Path, list[ActuatorDescription]]]:
    @derivation("controllable-building")
    def make_controllable_builder(input: Path):

        real_out = OUTPUT.get()
        with open(input, "rb") as f:
            json_obj = json.load(f)

        json_obj, hvac_actuators = make_unitary_hvac_controllable(json_obj)
        json_obj, baseboard_actuators = make_baseboard_controllable(json_obj)
        json_obj, fanonoff_actuators = make_fanonoff_controllable(json_obj)

        tmp_out = Path(tempfile.mkdtemp())

        # out.mkdir()

        json.dump(json_obj, open(tmp_out / "building.epjson", "w"), indent=4)
        json.dump(
            unstructure(hvac_actuators + baseboard_actuators + fanonoff_actuators),
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

    return parse_expr(make_controllable_builder(input_epjson))
