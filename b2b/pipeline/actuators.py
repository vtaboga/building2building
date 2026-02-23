import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

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
) -> str:
    """Create a constant schedule with the given type and value and return its name."""
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

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return self.actuators

    def zones(self) -> list[str]:
        return [self.zone]


def make_unitary_controllable(
    obj: dict[str, Any],
    *,
    epjson_type: str,
    fan_field: str,
    set_control_type: bool,
) -> tuple[dict[str, Any], list[UnitarySystem]]:
    """Generic discovery and mutation for any AirLoopHVAC unitary type that
    follows the one-zone-per-loop / ConstantVolume:NoReheat pattern.

    Parameters
    ----------
    epjson_type:
        The full EnergyPlus type string, e.g. "AirLoopHVAC:UnitarySystem".
    fan_field:
        The field name on the equipment object that holds the fan name.
        "supply_fan_name" for UnitarySystem, "supply_air_fan_name" for
        UnitaryHeatPump:AirToAir.
    set_control_type:
        Whether to write control_type = "SetPoint" onto the object.
        UnitarySystem requires it; UnitaryHeatPump:AirToAir does not have
        that field.
    """
    epjson_type_literal = rdflib.Literal(epjson_type)

    obj = deepcopy(obj)
    devices = []

    ont = Ontology.from_object(obj)
    g = ont.rdf

    setpoint_managers = obj.setdefault("SetpointManager:Scheduled", {})

    onoff_stl_name = create_onoff_availability_stl(
        obj, name="unitaryhvac fan availibiliby stl"
    )
    temp_stl_name = create_temp_stl(
        obj, 5.0, 50.0, name="unitaryhvac temperature setpoints stl"
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

    # Step 2: terminal_inlet_node -> splitter_inlet_node
    terminal_to_splitter_inlet: dict[str, str] = {}
    for row in g.query("""
        SELECT ?splitterInletNode ?outletNode
        WHERE {
            ?splitter a "AirLoopHVAC:ZoneSplitter" .
            ?splitter idf:inlet_node_name ?splitterInletNode .
            ?splitter idf:nodes ?head .
            ?head rdf:rest*/rdf:first ?item .
            ?item idf:outlet_node_name ?outletNode .
        }
    """):
        terminal_to_splitter_inlet[str(row.outletNode)] = str(row.splitterInletNode)

    # Step 3: splitter_inlet_node -> loop_name
    # demand_side_inlet_node_names can be either a direct node literal (VAV
    # buildings) or a NodeList name (UnitarySystem buildings) — handle both.
    splitter_inlet_to_loop: dict[str, str] = {
        str(row.demandInletNode): str(row.loop)
        for row in g.query("""
            SELECT ?loop ?demandInletNode
            WHERE {
                ?loop a "AirLoopHVAC" .
                ?loop idf:demand_side_inlet_node_names ?demandInletNode .
            }
        """)
    }
    for row in g.query("""
        SELECT ?loop ?nodeValue
        WHERE {
            ?loop a "AirLoopHVAC" .
            ?loop idf:demand_side_inlet_node_names ?nodeListName .
            ?nodeListName a "NodeList" .
            ?nodeListName idf:nodes ?head .
            ?head rdf:rest*/rdf:first ?item .
            ?item idf:node_name ?nodeValue .
        }
    """):
        splitter_inlet_to_loop[str(row.nodeValue)] = str(row.loop)

    # Step 4: loop_name -> (unitary_name, outlet_node)
    # Walk: AirLoopHVAC -> BranchList -> Branch -> component of the target type
    loop_to_unitary: dict[str, tuple[str, str]] = {
        str(row.loop): (str(row.unitaryName), str(row.outletNode))
        for row in g.query(
            """
            SELECT ?loop ?unitaryName ?outletNode
            WHERE {
                ?loop a "AirLoopHVAC" .
                ?loop idf:branch_list_name ?branchListName .
                ?branchListName a "BranchList" .
                ?branchListName idf:branches ?branchListHead .
                ?branchListHead rdf:rest*/rdf:first ?branchItem .
                ?branchItem idf:branch_name ?branchName .
                ?branchName a "Branch" .
                ?branchName idf:components ?componentsHead .
                ?componentsHead rdf:rest*/rdf:first ?comp .
                ?comp idf:component_object_type ?unitaryType .
                ?comp idf:component_name ?unitaryName .
                ?comp idf:component_outlet_node_name ?outletNode .
            }
        """,
            initBindings={"unitaryType": epjson_type_literal},
        )
    }

        # UnitaryHeatPump:AirToAir uses "supply_air_fan_name", while UnitarySystem
        # uses "supply_fan_name".
        if loop_type_name == "AirLoopHVAC:UnitaryHeatPump:AirToAir":
            supply_fan_name = unitary_system.get("supply_air_fan_name")
        else:
            supply_fan_name = unitary_system.get("supply_fan_name")

        if not isinstance(supply_fan_name, str) or not supply_fan_name.strip():
            # Keep going: setpoints can still be created even if fan name is missing.
            supply_fan_name = None

        system = obj[epjson_type][unitary_name]

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

        new_actuators = []

        fan_mode_schedule_name = create_schedule_constant(
            obj, onoff_stl_name, 1, name="unitaryhvac fan mode schedule"
        )
        system["supply_air_fan_operating_mode_schedule_name"] = fan_mode_schedule_name

        supply_fan_name = system[fan_field]
        new_actuators.append(
            ActuatorDescription(
                "Fan", "Fan Air Mass Flow Rate", supply_fan_name, "[kg/s]", 0, 100
            )
        )

        node_to_roles = {outlet_node: ["outlet"]}
        for node_name in sorted(node_to_roles):
            roles_str = "+".join(sorted(node_to_roles[node_name]))

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

        devices.append(UnitarySystem(zone, new_actuators))

    return obj, devices


def make_unitary_system_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], list[UnitarySystem]]:
    return make_unitary_controllable(
        obj,
        epjson_type="AirLoopHVAC:UnitarySystem",
        fan_field="supply_fan_name",
        set_control_type=True,
    )


def make_heat_pump_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], list[UnitarySystem]]:
    return make_unitary_controllable(
        obj,
        epjson_type="AirLoopHVAC:UnitaryHeatPump:AirToAir",
        fan_field="supply_air_fan_name",
        set_control_type=False,
    )


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
    """Find all baseboards (electric or water) via EquipmentConnections and
    expose their availability schedule as a controllable schedule."""

    obj = deepcopy(obj)
    ont = Ontology.from_object(obj)
    g = ont.rdf

    binary_stl = create_onoff_availability_stl(obj, name="baseboard availibility")

    baseboard_query = """
        SELECT ?zone ?baseboardName
        WHERE {
            ?equipConn a "ZoneHVAC:EquipmentConnections" .
            ?equipConn idf:zone_name ?zone .
            ?equipConn idf:zone_conditioning_equipment_list_name ?equipList .

            ?equipList a "ZoneHVAC:EquipmentList" .
            ?equipList idf:equipment ?equipHead .
            ?equipHead rdf:rest*/rdf:first ?equipItem .
            ?equipItem idf:zone_equipment_object_type ?baseboardType .
            ?equipItem idf:zone_equipment_name ?baseboardName .
        }
    """

    new_devices = []
    for baseboard_type in [
        "ZoneHVAC:Baseboard:Convective:Electric",
        "ZoneHVAC:Baseboard:Convective:Water",
    ]:
        type_literal = rdflib.Literal(baseboard_type)
        for row in g.query(
            baseboard_query, initBindings={"baseboardType": type_literal}
        ):
            baseboard_name = str(row.baseboardName)
            new_schedule_name = create_schedule_constant(
                obj,
                binary_stl,
                1,
                name="controllable schedule for baseboard",
            )
            obj[baseboard_type][baseboard_name]["availability_schedule_name"] = (
                new_schedule_name
            )
            new_devices.append(
                Baseboard(
                    ActuatorDescription(
                        component_type="Schedule:Constant",
                        control_type="Schedule Value",
                        component_name=new_schedule_name,
                        units="Availability",
                        lower_bound=0.0,
                        upper_bound=1.0,
                    )
                )
            )

    return obj, new_devices


@dataclass
class VAVTerminal:
    zone: str
    flow_fraction: ActuatorDescription
    heating_setpoint: ActuatorDescription
    cooling_setpoint: ActuatorDescription


@dataclass
class VAVSystem:
    supply_temp_setpoint: ActuatorDescription
    terminals: list[VAVTerminal]
    equipment_type: Literal["vavsystem"] = "vavsystem"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        out = [self.supply_temp_setpoint]
        for vav in self.terminals:
            out.append(vav.flow_fraction)
            out.append(vav.heating_setpoint)
            out.append(vav.cooling_setpoint)
        return out

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


def remove_thermostat_ems_overrides(obj: dict[str, Any]) -> None:
    """Remove EMS optimum-start programs that override thermostat setpoint
    schedules.

    ASHRAE 90.1 OfficeMedium buildings include EMS programs that SET
    CLGSETP_SCH / HTGSETP_SCH actuators at BeginTimestepBeforePredictor.
    These fight any external setpoint control and must be removed.

        # Create a new controllable schedule for the setpoint 
        sched_name = create_schedule_constant(
            obj,
            temp_stl_name,
            60.0,
            name=f"controllable setpoint for {wh_name}",
            gensym=gensym,
        )
        if any(act_name in lines for act_name in target_actuator_names):
            programs_to_remove.add(prog_name)

    # 3. Collect sensor / internal-variable names used by those programs.
    sensor_names: set[str] = set()
    ivar_names: set[str] = set()
    for prog_name in programs_to_remove:
        prog = ems_programs[prog_name]
        lines = " ".join(
            l.get("program_line", "") for l in prog.get("lines", [])
        )
        for sname in obj.get("EnergyManagementSystem:Sensor", {}):
            if sname in lines:
                sensor_names.add(sname)
        for ivname in obj.get("EnergyManagementSystem:InternalVariable", {}):
            if ivname in lines:
                ivar_names.add(ivname)

    # 4. Delete calling managers that reference removed programs.
    for pcm_name in list(
        obj.get("EnergyManagementSystem:ProgramCallingManager", {})
    ):
        pcm = obj["EnergyManagementSystem:ProgramCallingManager"][pcm_name]
        progs = [p.get("program_name", "") for p in pcm.get("programs", [])]
        if any(p in programs_to_remove for p in progs):
            del obj["EnergyManagementSystem:ProgramCallingManager"][pcm_name]

    # 5. Delete programs, actuators, sensors, internal variables.
    for prog_name in programs_to_remove:
        ems_programs.pop(prog_name, None)
    for act_name in target_actuator_names:
        ems_actuators.pop(act_name, None)
    for sname in sensor_names:
        obj.get("EnergyManagementSystem:Sensor", {}).pop(sname, None)
    for ivname in ivar_names:
        obj.get("EnergyManagementSystem:InternalVariable", {}).pop(ivname, None)

    # 6. If any top-level EMS dict is now empty, remove it.
    for ems_key in [
        "EnergyManagementSystem:Actuator",
        "EnergyManagementSystem:Program",
        "EnergyManagementSystem:ProgramCallingManager",
        "EnergyManagementSystem:Sensor",
        "EnergyManagementSystem:InternalVariable",
    ]:
        if ems_key in obj and not obj[ems_key]:
            del obj[ems_key]


def make_vav_system_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[VAVSystem]]:
    obj = deepcopy(obj)
    ontology = Ontology.from_object(obj)
    g = ontology.rdf

    # Remove EMS optimum-start programs before any mutations — they override
    # thermostat setpoint schedules and would fight our control.
    remove_thermostat_ems_overrides(obj)

    temp_stl_name = create_temp_stl(obj, 10.0, 55.0, name="vav supply temp stl")
    htg_stl_name = create_temp_stl(obj, 10.0, 35.0, name="vav heating setpoint stl")
    clg_stl_name = create_temp_stl(obj, 18.0, 40.0, name="vav cooling setpoint stl")
    setpoint_managers = obj.setdefault("SetpointManager:Scheduled", {})

    # Fraction STL for minimum air flow schedules (0-1, no unit type)
    fraction_stl_name = f"B2B vav min flow fraction stl ({gensym()})"
    obj.setdefault("ScheduleTypeLimits", {})[fraction_stl_name] = {
        "lower_limit_value": 0.0,
        "upper_limit_value": 1.0,
        "numeric_type": "Continuous",
    }

    # Step 1: zone -> (terminal_name, terminal_inlet_node)
    # Walk: EquipmentConnections -> EquipmentList -> ADU -> VAV:Reheat terminal
    zone_terminal_nodes: dict[str, tuple[str, str]] = {
        str(row.zone): (
            str(row.terminalName),
            str(row.terminalInletNode),
        )
        for row in g.query("""
            SELECT ?zone ?terminalName ?terminalInletNode
            WHERE {
                ?equipConn a "ZoneHVAC:EquipmentConnections" .
                ?equipConn idf:zone_name ?zone .
                ?equipConn idf:zone_conditioning_equipment_list_name ?equipList .

                ?equipList a "ZoneHVAC:EquipmentList" .
                ?equipList idf:equipment ?equipHead .
                ?equipHead rdf:rest*/rdf:first ?equipItem .
                ?equipItem idf:zone_equipment_name ?aduName .

                ?aduName a "ZoneHVAC:AirDistributionUnit" .
                ?aduName idf:air_terminal_name ?terminalName .

                ?terminalName a "AirTerminal:SingleDuct:VAV:Reheat" .
                ?terminalName idf:air_inlet_node_name ?terminalInletNode .
            }
        """)
    }

    # Step 2: terminal_inlet_node -> splitter_inlet_node
    terminal_to_loop_demand: dict[str, str] = {}
    for row in g.query("""
        SELECT ?splitterInletNode ?outletNode
        WHERE {
            ?splitter a "AirLoopHVAC:ZoneSplitter" .
            ?splitter idf:inlet_node_name ?splitterInletNode .
            ?splitter idf:nodes ?head .
            ?head rdf:rest*/rdf:first ?item .
            ?item idf:outlet_node_name ?outletNode .
        }
    """):
        terminal_to_loop_demand[str(row.outletNode)] = str(row.splitterInletNode)

    # Step 3: demand inlet node -> (loop name, supply outlet node)
    loop_by_demand: dict[str, tuple[str, str]] = {
        str(row.demandInletNode): (str(row.loop), str(row.supplyOutletNode))
        for row in g.query("""
            SELECT ?loop ?demandInletNode ?supplyOutletNode
            WHERE {
                ?loop a "AirLoopHVAC" .
                ?loop idf:demand_side_inlet_node_names ?demandInletNode .
                ?loop idf:supply_side_outlet_node_names ?supplyOutletNode .
            }
        """)
    }

    # Step 4: group zones by loop
    loops_dict: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for zone, (
        terminal_name,
        terminal_inlet,
    ) in zone_terminal_nodes.items():
        demand_node = terminal_to_loop_demand.get(terminal_inlet)
        if demand_node is None:
            continue
        loop_name, supply_outlet = loop_by_demand[demand_node]
        loops_dict.setdefault(loop_name, (supply_outlet, []))[1].append(
            (zone, terminal_name)
        )

    def install_supply_temp_actuator(supply_outlet: str) -> ActuatorDescription:
        """Remove any pre-existing setpoint managers targeting this node and
        replace with a controllable Schedule:Constant + SetpointManager:Scheduled.

        Pre-existing managers (of any type) would override an EMS node setpoint
        actuator every timestep, making it ineffective. Replacing them with our
        own scheduled manager gives us authoritative control.
        """
        for spm_type in list(obj.keys()):
            if not spm_type.startswith("SetpointManager:"):
                continue
            to_delete = [
                name
                for name, spm in obj[spm_type].items()
                if spm.get("setpoint_node_or_nodelist_name", "").upper()
                == supply_outlet.upper()
            ]
            for name in to_delete:
                del obj[spm_type][name]

        sched_name = create_schedule_constant(
            obj, temp_stl_name, 13, name="vav supply temp setpoint schedule"
        )
        spm_name = f"B2B VAV Supply Temp SPM for {supply_outlet} ({gensym()})"
        setpoint_managers[spm_name] = {
            "control_variable": "Temperature",
            "schedule_name": sched_name,
            "setpoint_node_or_nodelist_name": supply_outlet,
        }
        return ActuatorDescription(
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=sched_name,
            units="[C]",
            lower_bound=10.0,
            upper_bound=55.0,
        )

    def install_flow_fraction_actuator(terminal_name: str) -> ActuatorDescription:
        """Switch the VAV terminal to schedule-based minimum flow control and
        return a controllable flow fraction actuator.

        With zone_minimum_air_flow_input_method = "Constant", EnergyPlus clamps
        the damper at constant_minimum_air_flow_fraction regardless of what the
        RL agent requests.  Switching to "Scheduled" with a controllable schedule
        gives the agent full damper modulation range [0, 1].
        """
        terminal = obj["AirTerminal:SingleDuct:VAV:Reheat"][terminal_name]
        sched_name = create_schedule_constant(
            obj, fraction_stl_name, 0.3, name=f"vav flow fraction schedule"
        )
        terminal["zone_minimum_air_flow_input_method"] = "Scheduled"
        terminal["minimum_air_flow_fraction_schedule_name"] = sched_name

        return ActuatorDescription(
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=sched_name,
            units="[frac]",
            lower_bound=0.0,
            upper_bound=1.0,
        )

    def install_thermostat_actuators(
        zone: str,
    ) -> tuple[ActuatorDescription, ActuatorDescription]:
        """Replace thermostat setpoint schedules for a zone with controllable
        Schedule:Constant objects and return (heating, cooling) actuators.
        """
        # Find the ZoneControl:Thermostat for this zone
        thermostat_controls = obj.get("ZoneControl:Thermostat", {})
        tc = None
        for _name, candidate in thermostat_controls.items():
            if candidate.get("zone_or_zonelist_name") == zone:
                tc = candidate
                break

        if tc is None:
            raise ValueError(f"No ZoneControl:Thermostat found for zone {zone}")

        dsp_name = tc["control_1_name"]
        dsp = obj["ThermostatSetpoint:DualSetpoint"][dsp_name]

        htg_sched = create_schedule_constant(
            obj, htg_stl_name, 21, name=f"vav htg setpoint {zone}"
        )
        clg_sched = create_schedule_constant(
            obj, clg_stl_name, 24, name=f"vav clg setpoint {zone}"
        )

        dsp["heating_setpoint_temperature_schedule_name"] = htg_sched
        dsp["cooling_setpoint_temperature_schedule_name"] = clg_sched

        htg_actuator = ActuatorDescription(
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=htg_sched,
            units="[C]",
            lower_bound=10.0,
            upper_bound=35.0,
        )
        clg_actuator = ActuatorDescription(
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=clg_sched,
            units="[C]",
            lower_bound=18.0,
            upper_bound=40.0,
        )

        return htg_actuator, clg_actuator

    loops = []
    for loop_name, (supply_outlet, zone_terminals) in loops_dict.items():
        terminals = []
        for zone, terminal_name in zone_terminals:
            flow_act = install_flow_fraction_actuator(terminal_name)
            htg_act, clg_act = install_thermostat_actuators(zone)
            terminals.append(
                VAVTerminal(
                    zone=zone,
                    flow_fraction=flow_act,
                    heating_setpoint=htg_act,
                    cooling_setpoint=clg_act,
                )
            )
        loops.append(
            VAVSystem(
                supply_temp_setpoint=install_supply_temp_actuator(supply_outlet),
                terminals=terminals,
            )
        )

    return obj, loops


# The purpose of this type (compared to types.Equipment) is to actually list all
# the different types of things an `Equipment` can be. If we don't do that, the
# cattrs library cant rehydrate the dataclasses correctly.
AnyEquipment = VAVSystem | UnitarySystem | Baseboard


def make_all_equipment(
    json_obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[AnyEquipment]]:
    all_functions = [
        make_unitary_system_controllable,
        make_heat_pump_controllable,
        make_vav_system_controllable,
        make_baseboard_controllable,
    ]
    all_equipment: list[AnyEquipment] = []
    for func in all_functions:
        json_obj, devices = func(json_obj)
        all_equipment += devices

    return json_obj, all_equipment


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

        json_obj, equipment = make_all_equipment(json_obj)

        tmp_out = Path(tempfile.mkdtemp())

        json.dump(json_obj, open(tmp_out / "building.epjson", "w"), indent=4)
        json.dump(
            unstructure(all_actuators),
            open(tmp_out / "actuators.json", mode="w"),
            indent=4,
        )

        shutil.move(tmp_out, real_out)

    @expression()
    def parse_expr(folder: Path) -> tuple[Path, Sequence[Equipment]]:
        with open(folder / "equipment.json", mode="r") as f:
            actuators_json = json.load(f)

        return folder / "building.epjson", structure(actuators_json, list[AnyEquipment])

    return parse_expr(make_controllable_builder(input_epjson, selected_controls))
