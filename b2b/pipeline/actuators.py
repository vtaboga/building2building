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
    """Create a constant schedule with the given type and value and return its name."""
    schedule_constants = obj.setdefault("Schedule:Constant", {})
    name = f"B2B {name} ({gensym()})"
    schedule_constants[name] = {
        "hourly_value": hourly_value,
        "schedule_type_limits_name": stl_name,
    }
    return name


@dataclass
class UnitarySystem:
    zone: str
    actuators: list[ActuatorDescription]
    equipment_type: Literal["unitarysystem"] = "unitarysystem"

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

    # Step 1: zone -> terminal_inlet_node
    # Walk: EquipmentConnections -> EquipmentList -> ADU -> NoReheat terminal
    zone_to_terminal_inlet: dict[str, str] = {
        str(row.zone): str(row.terminalInletNode)
        for row in g.query("""
            SELECT ?zone ?terminalInletNode
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

                ?terminalName a "AirTerminal:SingleDuct:ConstantVolume:NoReheat" .
                ?terminalName idf:air_inlet_node_name ?terminalInletNode .
            }
        """)
    }

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

    # Assemble and mutate
    for zone, terminal_inlet in zone_to_terminal_inlet.items():
        splitter_inlet = terminal_to_splitter_inlet.get(terminal_inlet)
        if splitter_inlet is None:
            continue
        loop_name = splitter_inlet_to_loop.get(splitter_inlet)
        if loop_name is None:
            continue
        unitary_entry = loop_to_unitary.get(loop_name)
        if unitary_entry is None:
            continue
        unitary_name, outlet_node = unitary_entry

        system = obj[epjson_type][unitary_name]

        if set_control_type:
            system["control_type"] = "SetPoint"

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
    mass_flow_setpoint: ActuatorDescription
    mass_flow_min_available: ActuatorDescription
    mass_flow_max_available: ActuatorDescription


@dataclass
class VAVSystem:
    supply_temp_setpoint: ActuatorDescription
    terminals: list[VAVTerminal]
    equipment_type: Literal["vavsystem"] = "vavsystem"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        out = [self.supply_temp_setpoint]
        for vav in self.terminals:
            out.append(vav.mass_flow_setpoint)
            out.append(vav.mass_flow_min_available)
            out.append(vav.mass_flow_max_available)
        return out

    def zones(self) -> list[str]:
        return [vav.zone for vav in self.terminals]


def make_vav_system_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[VAVSystem]]:
    ontology = Ontology.from_object(obj)
    g = ontology.rdf

    # Step 1: zone -> (terminal_inlet_node, terminal_outlet_node)
    zone_terminal_nodes: dict[str, tuple[str, str]] = {
        str(row.zone): (str(row.terminalInletNode), str(row.terminalOutletNode))
        for row in g.query("""
            SELECT ?zone ?terminalInletNode ?terminalOutletNode
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
                ?terminalName idf:air_outlet_node_name ?terminalOutletNode .
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
    for zone, (terminal_inlet, terminal_outlet) in zone_terminal_nodes.items():
        demand_node = terminal_to_loop_demand.get(terminal_inlet)
        if demand_node is None:
            continue
        loop_name, supply_outlet = loop_by_demand[demand_node]
        loops_dict.setdefault(loop_name, (supply_outlet, []))[1].append(
            (zone, terminal_outlet)
        )

    def flow_actuator(node: str, control_type: str) -> ActuatorDescription:
        return ActuatorDescription(
            component_type="System Node Setpoint",
            control_type=control_type,
            component_name=node,
            units="[kg/s]",
            lower_bound=0.0,
            upper_bound=1.0,
        )

    loops = []
    for loop_name, (supply_outlet, zone_terminals) in loops_dict.items():
        terminals = [
            VAVTerminal(
                zone=zone,
                mass_flow_setpoint=flow_actuator(
                    outlet.upper(), "Mass Flow Rate Setpoint"
                ),
                mass_flow_min_available=flow_actuator(
                    outlet.upper(), "Mass Flow Rate Minimum Available Setpoint"
                ),
                mass_flow_max_available=flow_actuator(
                    outlet.upper(), "Mass Flow Rate Maximum Available Setpoint"
                ),
            )
            for zone, outlet in zone_terminals
        ]
        loops.append(
            VAVSystem(
                supply_temp_setpoint=ActuatorDescription(
                    component_type="System Node Setpoint",
                    control_type="Temperature Setpoint",
                    component_name=supply_outlet.upper(),
                    units="[C]",
                    lower_bound=10.0,
                    upper_bound=45.0,
                ),
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
