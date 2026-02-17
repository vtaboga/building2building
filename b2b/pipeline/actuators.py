import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import rdflib
from cattrs import structure, unstructure
from minergym.ontology import Ontology

from b2b.store import (
    OUTPUT,
    Expression,
    Realizable,
    derivation,
    expression,
)
from b2b.types import ActuatorDescription, Equipment


@dataclass
class Gensym:
    i: int = 0

    def __call__(self) -> int:
        out = self.i
        self.i += 1
        return out

    def reset(self):
        self.i = 0


# global counter for schedule type limits and constants
# Use this to ensure that the names are unique across all components.
gensym = Gensym()


def create_onoff_availability_stl(obj: dict[str, Any], *, name: str = "OnOff") -> str:
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


temp_stl_lower_bound = 5.0
temp_stl_upper_bound = 50.0


def create_temp_stl(
    obj: dict[str, Any], lower: float, upper: float, *, name: str = "Temperature"
) -> str:
    """Create a continuous ScheduleTypeLimits for temperatures and return its
    name.

    """
    schedule_type_limits = obj.setdefault("ScheduleTypeLimits", {})

    name = f"B2B {name} ({gensym()})"
    schedule_type_limits[name] = {
        "lower_limit_value": lower,
        "upper_limit_value": upper,
        "numeric_type": "Continuous",
        "unit_type": "Temperature",
    }

    return name


def create_schedule_constant(
    obj: dict[str, Any],
    stl_name: str,
    hourly_value: int,
    *,
    name: str = "constant schedule",
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


@dataclass
class AirloopHVAC:
    zone: str

    actuators: list[ActuatorDescription]

    equipment_type: Literal["airloophvac"] = "airloophvac"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return self.actuators

    def zones(self) -> list[str]:
        return [self.zone]


def make_airloophvac_controllable(
    obj: dict[str, Any],
    type: str,
) -> tuple[dict[str, Any], list[AirloopHVAC]]:
    """Find all "AirLoopHVAC:UnitaryHeatPump:AirToAir" and expose the relevant
    node setpoints as schedules that can be controlled by minergym.

    This is done in many steps:

    1. We create the relevant schedule type descriptors (ScheduleTypeLimits)
       that will be used by all generated schedules. Those consist of

       1. an OnOff type which will be used by the system fan's schedule
          ("supply_air_fan_operating_mode_schedule_name").

       2. a Temperature type which will be used by all schedules we use for
          controlling temperature.

    2. We query the ontology and look for all "AirLoopHVAC:{type}".
       For each of those, we do the following:

       1. We set the control_type to SetPoint

       2. We create a schedule to make the fan mode controllable.

       3. For each of the nodes associated to system (outlet_node,
          cooling_coil_node, heating_coil_node, supplemental_coil_node), we set
          up a scheduler and a setpoint manager that makes that specific node
          controllable.

    """

    complete_type = f"AirLoopHVAC:{type}"
    complete_type_rdf = rdflib.Literal(complete_type)

    obj = deepcopy(obj)
    devices = []

    ont = Ontology.from_object(obj)

    setpoint_managers = obj.setdefault("SetpointManager:Scheduled", {})

    # We define the schedule type descriptors (onoff and temperature)

    onoff_stl_name = create_onoff_availability_stl(
        obj, name="unitaryhvac fan availibiliby stl"
    )
    temp_stl_name = create_temp_stl(
        obj,
        5.0,
        50.0,
        name="unitaryhvac temperature setpoints stl",
    )

    # Note: I wrapped each coil section in OPTIONAL blocks because not all
    # unitary systems have all three coil types (e.g., cooling-only systems
    # won't have heating coils).
    #
    # TODO: actually handle cases where some of these are None.
    all_loops_query = """# -*- mode: sparql-*-
SELECT ?loop ?zone ?outlet_node
WHERE {
  ?loop a ?completeType .
  ?loop idf:air_outlet_node_name ?outlet_node .
  ?loop idf:controlling_zone_or_thermostat_location ?zone .
}


"""
    # loop, outlet, cooling_coil, cooling_coil_node
    for (
        loop,
        zone,
        outlet_node,
    ) in ont.rdf.query(
        all_loops_query,
        initBindings={
            "completeType": complete_type_rdf,
        },
    ):
        # new_obj = AirloopHVAC()
        system = obj[complete_type][str(loop)]

        system["control_type"] = "SetPoint"

        new_actuators = []

        # Set up the fan mode. It should be always on
        fan_mode_schedule_name = create_schedule_constant(
            obj,
            onoff_stl_name,
            1,
            name="unitaryhvac fan mode schedule",
        )

        system["supply_air_fan_operating_mode_schedule_name"] = fan_mode_schedule_name

        supply_fan_name = system["supply_fan_name"]

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

        node_to_roles = {outlet_node.toPython(): ["outlet"]}
        for node_name in sorted(node_to_roles):
            roles = sorted(node_to_roles[node_name])
            roles_str = "+".join(roles)

            sched_constant_name = create_schedule_constant(
                obj,
                temp_stl_name,
                22,
                name=f"unitaryhvac {roles_str} temp setpoint schedule",
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

        devices.append(
            AirloopHVAC(
                zone,
                new_actuators,
            )
        )

    return obj, devices


@dataclass
class Baseboard:
    actuator: ActuatorDescription

    equipment_type: Literal["baseboard"] = "baseboard"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.actuator]

    def zones(self) -> list[str]:
        return []


def make_baseboard_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[Baseboard]]:
    """Find all baseboards and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(obj, name="baseboard availibility")

    all_baseboards_query = """# -*- mode: sparql -*-
SELECT ?baseboard WHERE {
  ?baseboard a "ZoneHVAC:Baseboard:Convective:Electric" .
}"""

    new_devices = []
    for (baseboard,) in ont.rdf.query(all_baseboards_query):
        baseboard_name = baseboard.toPython()
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name="controllable schedule for baseboard",
        )

        obj["ZoneHVAC:Baseboard:Convective:Electric"][baseboard_name][
            "availability_schedule_name"
        ] = new_schedule_name

        new_devices.append(
            Baseboard(
                ActuatorDescription(
                    component_type="Schedule:Constant",
                    control_type="Schedule Value",
                    component_name=new_schedule_name,
                    units="Availability",
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            )
        )

    return obj, new_devices


@dataclass
class FanOnOff:
    actuator: ActuatorDescription

    equipment_type: Literal["fanonoff"] = "fanonoff"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.actuator]

    def zones(self) -> list[str]:
        return []


def make_fanonoff_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[FanOnOff]]:
    """Find all Fan:OnOff objects and expose the availability schedule as a
    schedule that can be controlled."""

    obj = deepcopy(obj)

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(obj, name="fanonoff availability")

    all_fans_query = """# -*- mode: sparql -*-
SELECT ?fan WHERE {
  ?fan a "Fan:OnOff" .
}"""

    new_devices = []

    for (fan,) in ont.rdf.query(all_fans_query):
        fan_name = fan.toPython()
        new_schedule_name = create_schedule_constant(
            obj,
            binary_stl,
            1,
            name="controllable schedule for fanonoff",
        )

        obj["Fan:OnOff"][fan_name]["availability_schedule_name"] = new_schedule_name

        new_devices.append(
            FanOnOff(
                ActuatorDescription(
                    component_type="Schedule:Constant",
                    control_type="Schedule Value",
                    component_name=new_schedule_name,
                    units="Availability",
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            )
        )

    return obj, new_devices


# WaterHeater and friends dosn't seem to work. The actuator name that is
# produced is not present in the discovery edd and causes a runtime crash.
@dataclass
class WaterHeater:
    actuator: ActuatorDescription

    equipment_type: Literal["waterheater"] = "waterheater"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.actuator]

    def zones(self) -> list[str]:
        return []


def make_waterheater_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[WaterHeater]]:
    """Find all waterheaters and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    low, high = 5, 50

    obj = deepcopy(obj)

    ont = Ontology.from_object(obj)

    # those are the bound of water heater
    temp_stl_name = create_temp_stl(obj, low, high, name="water heater temperature stl")

    # SPARQL query for Water Heaters
    all_waterheaters_query = """# -*- mode: sparql -*-
    SELECT ?wh WHERE {
      ?wh a "WaterHeater:Mixed" .
    }"""

    new_devices = []

    for (wh_id,) in ont.rdf.query(all_waterheaters_query):
        wh_name = wh_id.toPython()
        wh_entry = obj["WaterHeater:Mixed"][wh_name]

        # Create a new controllable schedule for the setpoint
        sched_name = create_schedule_constant(
            obj,
            temp_stl_name,
            25,
            name=f"controllable setpoint for {wh_name}",
        )

        # Override the original schedule
        wh_entry["setpoint_temperature_schedule_name"] = sched_name

        new_devices.append(
            WaterHeater(
                ActuatorDescription(
                    component_type="WaterHeater",
                    control_type="Setpoint Temperature",
                    component_name=sched_name,
                    units="Temperature",
                    lower_bound=low,
                    upper_bound=high,
                ),
            )
        )

    return obj, new_devices


@dataclass
class Pump:
    actuator: ActuatorDescription

    equipment_type: Literal["pump"] = "pump"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.actuator]

    def zones(self) -> list[str]:
        return []


def make_pump_controllable(
    obj: dict[str, Any],
    *,
    gensym: Gensym | None = None,
) -> tuple[dict[str, Any], Sequence[Pump]]:
    """Find all Pump:ConstantSpeed and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)

    new_devices = []

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(obj, name="pump availability stl")

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
        )

        # Set the pump to use this new schedule (Adding field if not present)
        obj["Pump:ConstantSpeed"][pump_name]["pump_scheduling_control_scheme"] = (
            "Schedule"
        )
        obj["Pump:ConstantSpeed"][pump_name]["availability_schedule_name"] = (
            new_schedule_name
        )

        new_devices.append(
            Pump(
                ActuatorDescription(
                    component_type="Schedule:Constant",
                    control_type="Schedule Value",
                    component_name=new_schedule_name,
                    units="Availability",
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            )
        )

    return obj, new_devices


@dataclass
class AirTerminal:
    actuator: ActuatorDescription

    equipment_type: Literal["airterminal"] = "airterminal"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.actuator]

    def zones(self) -> list[str]:
        return []


def make_airterminal_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[AirTerminal]]:
    """Find all ConstantVolume:NoReheat Air Terminals and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)

    new_devices = []

    ont = Ontology.from_object(obj)

    binary_stl = create_onoff_availability_stl(obj, name="terminal availability stl")

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
        )

        obj["AirTerminal:SingleDuct:ConstantVolume:NoReheat"][term_name][
            "availability_schedule_name"
        ] = new_schedule_name

        new_devices.append(
            AirTerminal(
                ActuatorDescription(
                    component_type="Schedule:Constant",
                    control_type="Schedule Value",
                    component_name=new_schedule_name,
                    units="Availability",
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            )
        )

    return obj, new_devices


@dataclass
class OutdoorAir:
    actuator: ActuatorDescription

    equipment_type: Literal["outdoorair"] = "outdoorair"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.actuator]

    def zones(self) -> list[str]:
        return []


def make_controller_outdoorair_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[OutdoorAir]]:
    """Find all Controller:OutdoorAir and for each of those, expose the availibility
    schedue as a schedule that can be controlled."""

    obj = deepcopy(obj)

    new_devices = []

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
        )

        obj["Controller:OutdoorAir"][ctrl_name]["minimum_outdoor_air_schedule_name"] = (
            new_schedule_name
        )

        new_devices.append(
            OutdoorAir(
                ActuatorDescription(
                    component_type="Schedule:Constant",
                    control_type="Schedule Value",
                    component_name=new_schedule_name,
                    units="Fraction",
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            )
        )

    return obj, new_devices


# The purpose of this type (compared to types.Equipment) is to actually list all
# the different types of things an `Equipment` can be. If we don't do that, the
# cattrs library cant rehydrate the dataclasses correctly.
AnyEquipment = AirloopHVAC | Baseboard | FanOnOff | AirTerminal | OutdoorAir


def make_all_equipment(
    json_obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[AnyEquipment]]:
    all_functions = [
        lambda obj: make_airloophvac_controllable(obj, "UnitarySystem"),
        make_baseboard_controllable,
        # make_fanonoff_controllable,
        # make_waterheater_controllable,
        # make_pump_controllable,
        # make_airterminal_controllable,
        # make_controller_outdoorair_controllable,
    ]
    all_equipment: list[AnyEquipment] = []
    for func in all_functions:
        json_obj, devices = func(json_obj)
        all_equipment += devices

    return json_obj, all_equipment


def make_controllable(
    input_epjson: Realizable,
) -> Expression[tuple[Path, Sequence[Equipment]]]:
    @derivation("controllable-building")
    def make_controllable_builder(input: Path):
        real_out = OUTPUT.get()
        with open(input, "rb") as f:
            json_obj = json.load(f)

        gensym.reset()

        json_obj, equipment = make_all_equipment(json_obj)

        tmp_out = Path(tempfile.mkdtemp())

        json.dump(json_obj, open(tmp_out / "building.epjson", "w"), indent=4)
        json.dump(
            unstructure(equipment),
            open(tmp_out / "equipment.json", mode="w"),
            indent=4,
        )

        shutil.move(tmp_out, real_out)

    @expression()
    def parse_expr(folder: Path) -> tuple[Path, Sequence[Equipment]]:
        with open(folder / "equipment.json", mode="r") as f:
            actuators_json = json.load(f)

        return folder / "building.epjson", structure(actuators_json, list[AnyEquipment])

    return parse_expr(make_controllable_builder(input_epjson))
