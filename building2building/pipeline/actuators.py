import json
import logging
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import rdflib
from cattrs import structure, unstructure
from minergym.ontology import Ontology

from building2building.store import (
    OUTPUT,
    Expression,
    Realizable,
    derivation,
    expression,
)
from building2building.types import ActuatorDescription, Equipment


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
temp_stl_upper_bound = 50.0  # fallback; overridden per-system when possible

STD_AIR_DENSITY = 1.2  # kg/m³ at ~20 °C, 101.325 kPa
DEFAULT_FAN_MAX_KGS = 15.0  # fallback when design data is unavailable
DEFAULT_SAT_MAX_C = (
    60.0  # fallback when maximum_supply_air_temperature is Autosize or missing
)
# Upper bound on the ``Controller:OutdoorAir × Air Mass Flow Rate`` EMS
# actuator (kg/s). 5.0 kg/s exceeds the DOE OfficeMedium per-loop design
# supply-air flow (~4-6 kg/s) while staying well below the unphysical
# range.
OA_MASS_FLOW_MAX_KGS = 5.0


@dataclass(frozen=True)
class Bounds:
    """An inclusive ``(low, high)`` actuator range."""

    low: float
    high: float


# Static actuator setpoint bounds: the canonical, building-independent envelope
# for each controllable schedule. Unlike the fan/SAT bounds above (read from
# per-building design data, with the DEFAULT_* values as fallback), these never
# vary by building. building2building.morphology imports them for the matching
# NodeTypes so the morphological universe and the pipeline cannot drift apart.
VAV_SUPPLY_TEMP_C = Bounds(10.0, 55.0)   # SetpointManager:Scheduled supply-air temp
VAV_HEATING_SP_C = Bounds(10.0, 35.0)    # VAV DualSetpoint heating schedule
VAV_COOLING_SP_C = Bounds(18.0, 40.0)    # VAV DualSetpoint cooling schedule
FLOW_FRACTION = Bounds(0.0, 1.0)         # VAV minimum-air-flow fraction schedule
OA_MASS_FLOW_KGS = Bounds(0.0, OA_MASS_FLOW_MAX_KGS)  # Controller:OutdoorAir mass flow
HEATING_ONLY_SP_C = Bounds(10.0, 35.0)   # heating-only zone setpoint schedule

logger = logging.getLogger(__name__)

DX_COOLING_COMPRESSOR_MIN_OAT_C = 10.0


def _set_dx_cooling_compressor_lockout(
    obj: dict[str, Any],
    cooling_coil_type: str,
    cooling_coil_name: str,
    min_oat_c: float = DX_COOLING_COMPRESSOR_MIN_OAT_C,
) -> None:
    """Prevent DX cooling compressor operation below *min_oat_c* outdoor air.

    Without this, SetPoint-controlled UnitarySystem will activate the DX
    cooling coil whenever the SAT setpoint is below return air temperature
    — even in winter.  Running a DX cooling compressor at sub-zero outdoor
    temperatures causes frost/freeze and wastes energy.
    """
    dx_types = (
        "Coil:Cooling:DX:SingleSpeed",
        "Coil:Cooling:DX:TwoSpeed",
        "Coil:Cooling:DX:MultiSpeed",
        "Coil:Cooling:DX:TwoStageWithHumidityControlMode",
    )
    if cooling_coil_type not in dx_types:
        return
    coils = obj.get(cooling_coil_type, {})
    target = cooling_coil_name.upper()
    for name, coil in coils.items():
        if name == cooling_coil_name or name.upper() == target:
            coil["minimum_outdoor_dry_bulb_temperature_for_compressor_operation"] = (
                min_oat_c
            )
            return


def _read_fan_design_flow_kgs(obj: dict[str, Any], fan_name: str) -> float | None:
    """Read the fan's design max air flow [m³/s] from the epJSON and convert to kg/s.

    EnergyPlus object names are case-insensitive, so we fall back to a
    case-insensitive lookup when the exact key isn't found.
    """
    target = fan_name.upper()
    for fan_type in ("Fan:SystemModel", "Fan:OnOff", "Fan:ConstantVolume"):
        fans = obj.get(fan_type, {})
        fan = fans.get(fan_name)
        if fan is None:
            for k, v in fans.items():
                if k.upper() == target:
                    fan = v
                    break
        if fan is None:
            continue
        for key in ("design_maximum_air_flow_rate", "maximum_flow_rate"):
            val = fan.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val) * STD_AIR_DENSITY
    return None


def _read_max_supply_air_temp_c(
    obj: dict[str, Any], system_name: str, epjson_type: str
) -> float | None:
    """Try to read the maximum supply air temperature [°C] from the epJSON."""
    systems = obj.get(epjson_type, {})
    system = systems.get(system_name, {})
    val = system.get("maximum_supply_air_temperature")
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return None


def _set_fan_always_on(
    obj: dict[str, Any], fan_name: str, always_on_sched: str
) -> None:
    """Override a fan's availability schedule to *always_on_sched*.

    OfficeSmall (and similar) buildings use ``HVACOperationSchd`` which turns
    the fan off nights/weekends.  For RL control the fan must be always
    available so the agent can modulate mass flow at any time.
    """
    target = fan_name.upper()
    for fan_type in (
        "Fan:SystemModel",
        "Fan:OnOff",
        "Fan:ConstantVolume",
        "Fan:VariableVolume",
    ):
        fans = obj.get(fan_type, {})
        fan = fans.get(fan_name)
        if fan is None:
            for k, v in fans.items():
                if k.upper() == target:
                    fan = v
                    break
        if fan is not None:
            fan["availability_schedule_name"] = always_on_sched
            return


def _ensure_always_on_availability(
    obj: dict[str, Any], loop_name: str, always_on_sched: str
) -> None:
    """Replace NightCycle availability managers on *loop_name* with always-on.

    ``AvailabilityManager:NightCycle`` cycles the fan based on thermostat
    tolerance when the HVAC schedule says OFF.  Under RL control, equipment
    must be unconditionally available; replace with
    ``AvailabilityManager:Scheduled`` pointing at an always-on schedule.
    """
    loop_obj = obj.get("AirLoopHVAC", {}).get(loop_name)
    if loop_obj is None:
        return
    avail_list_name = loop_obj.get("availability_manager_list_name")
    if not avail_list_name:
        return
    avail_list = obj.get("AvailabilityManagerAssignmentList", {}).get(avail_list_name)
    if avail_list is None:
        return

    night_cycle_mgrs = obj.get("AvailabilityManager:NightCycle", {})
    scheduled_mgrs = obj.setdefault("AvailabilityManager:Scheduled", {})

    for mgr in avail_list.get("managers", []):
        if (
            mgr.get("availability_manager_object_type")
            != "AvailabilityManager:NightCycle"
        ):
            continue
        old_name = mgr.get("availability_manager_name", "")
        night_cycle_mgrs.pop(old_name, None)

        new_name = f"B2B Always On Avail for {loop_name} ({gensym()})"
        scheduled_mgrs[new_name] = {"schedule_name": always_on_sched}

        mgr["availability_manager_name"] = new_name
        mgr["availability_manager_object_type"] = "AvailabilityManager:Scheduled"

    if (
        "AvailabilityManager:NightCycle" in obj
        and not obj["AvailabilityManager:NightCycle"]
    ):
        del obj["AvailabilityManager:NightCycle"]


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


@dataclass
class HeatPump:
    """Equipment descriptor for an AirLoopHVAC:UnitaryHeatPump:AirToAir zone.

    Load-based heat pumps are controlled via thermostat setpoint schedules
    rather than supply air temperature setpoints.
    """

    zone: str
    heating_setpoint: ActuatorDescription
    cooling_setpoint: ActuatorDescription
    equipment_type: Literal["heatpump"] = "heatpump"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.heating_setpoint, self.cooling_setpoint]

    def zones(self) -> list[str]:
        return [self.zone]


def make_unitary_controllable(
    obj: dict[str, Any],
    *,
    epjson_type: str,
    fan_field: str,
    set_control_type: bool,
) -> tuple[dict[str, Any], list[UnitarySystem]]:
    """Generic discovery and mutation for AirLoopHVAC:UnitarySystem objects
    following the one-zone-per-loop / ConstantVolume:NoReheat pattern.

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

    # Step 1: zone -> terminal_inlet_node
    # Walk: EquipmentConnections -> EquipmentList -> ADU -> NoReheat terminal
    zone_to_terminal_inlet: dict[str, str] = {
        str(row.zone): str(row.terminalInletNode) for row in g.query("""
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
        str(row.demandInletNode): str(row.loop) for row in g.query("""
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

    # Step 5: loop_name -> demand_side_inlet_node
    # Needed to clean up SingleZone SPMs that target the demand inlet.
    loop_to_demand_inlet: dict[str, str] = {
        str(row.loop): str(row.demandInletNode) for row in g.query("""
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
        loop_to_demand_inlet[str(row.loop)] = str(row.nodeValue)

    always_on_sched_name = create_schedule_constant(
        obj, onoff_stl_name, 1, name="unitaryhvac always on availability"
    )

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

        cooling_coil_type = system.get("cooling_coil_object_type", "")
        cooling_coil_name = system.get("cooling_coil_name", "")
        if cooling_coil_type and cooling_coil_name:
            _set_dx_cooling_compressor_lockout(
                obj, cooling_coil_type, cooling_coil_name
            )

        new_actuators = []

        fan_mode_schedule_name = create_schedule_constant(
            obj, onoff_stl_name, 1, name="unitaryhvac fan mode schedule"
        )
        system["supply_air_fan_operating_mode_schedule_name"] = fan_mode_schedule_name

        supply_fan_name = system[fan_field]

        _set_fan_always_on(obj, supply_fan_name, always_on_sched_name)
        _ensure_always_on_availability(obj, loop_name, always_on_sched_name)

        design_kgs = _read_fan_design_flow_kgs(obj, supply_fan_name)
        if design_kgs is None:
            logger.warning(
                "Could not read design fan flow rate for fan %r; "
                "falling back to DEFAULT_FAN_MAX_KGS=%.1f kg/s",
                supply_fan_name,
                DEFAULT_FAN_MAX_KGS,
            )
        fan_upper_kgs = design_kgs if design_kgs is not None else DEFAULT_FAN_MAX_KGS

        new_actuators.append(
            ActuatorDescription(
                "Fan",
                "Fan Air Mass Flow Rate",
                supply_fan_name,
                "[kg/s]",
                0,
                fan_upper_kgs,
            )
        )

        sat_max = _read_max_supply_air_temp_c(obj, unitary_name, epjson_type)
        if sat_max is None:
            logger.warning(
                "Could not read maximum_supply_air_temperature for %s %r; "
                "falling back to DEFAULT_SAT_MAX_C=%.1f °C",
                epjson_type,
                unitary_name,
                DEFAULT_SAT_MAX_C,
            )
            sat_max = DEFAULT_SAT_MAX_C
        temp_stl_name = create_temp_stl(
            obj,
            temp_stl_lower_bound,
            sat_max,
            name="unitaryhvac temperature setpoints stl",
        )

        # Remove pre-existing setpoint managers on the supply outlet node
        # so they don't conflict with our scheduled manager.  Demand-side
        # SingleZone SPMs are left in place: with control_type="SetPoint"
        # the UnitarySystem reads only the outlet node setpoint, and the
        # existing zone thermostat SPMs satisfy EnergyPlus's
        # checkSetpointNodesAtEnd validation on the demand inlet.
        outlet_upper = outlet_node.upper()
        for spm_type in list(obj.keys()):
            if not spm_type.startswith("SetpointManager:"):
                continue
            to_delete = [
                name
                for name, spm in obj[spm_type].items()
                if spm.get("setpoint_node_or_nodelist_name", "").upper() == outlet_upper
            ]
            for name in to_delete:
                del obj[spm_type][name]

        sched_constant_name = create_schedule_constant(
            obj,
            temp_stl_name,
            22,
            name="unitaryhvac temp setpoint schedule",
        )
        spm_label = f"B2B Unitary TEMP SPM for {outlet_node} ({gensym()})"
        setpoint_managers[spm_label] = {
            "control_variable": "Temperature",
            "schedule_name": sched_constant_name,
            "setpoint_node_or_nodelist_name": outlet_node,
        }

        new_actuators.append(
            ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=sched_constant_name,
                units="Temperature",
                lower_bound=temp_stl_lower_bound,
                upper_bound=sat_max,
            )
        )

        devices.append(UnitarySystem(zone, new_actuators))

    return obj, devices


def convert_heat_pumps_to_unitary_systems(obj: dict[str, Any]) -> dict[str, Any]:
    """Convert AirLoopHVAC:UnitaryHeatPump:AirToAir objects to
    AirLoopHVAC:UnitarySystem with control_type="SetPoint".

    UnitaryHeatPump is load-based and ignores outlet node SAT setpoints.
    UnitarySystem with SetPoint control actively modulates coils to meet them,
    which is required for the two-actuator (fan flow + SAT) RL strategy.

    Must be called before make_unitary_system_controllable so that the
    converted systems are discovered by the SPARQL queries.
    """
    HP_TYPE = "AirLoopHVAC:UnitaryHeatPump:AirToAir"
    US_TYPE = "AirLoopHVAC:UnitarySystem"

    heat_pumps = obj.get(HP_TYPE, {})
    if not heat_pumps:
        return obj

    unitary_systems = obj.setdefault(US_TYPE, {})

    FIELD_RENAME: dict[str, str] = {
        "supply_air_fan_name": "supply_fan_name",
        "supply_air_fan_object_type": "supply_fan_object_type",
        "maximum_supply_air_temperature_from_supplemental_heater": (
            "maximum_supply_air_temperature"
        ),
    }

    FLOW_RATE_METHOD_FIELDS: dict[str, str] = {
        "cooling_supply_air_flow_rate": "cooling_supply_air_flow_rate_method",
        "heating_supply_air_flow_rate": "heating_supply_air_flow_rate_method",
        "no_load_supply_air_flow_rate": "no_load_supply_air_flow_rate_method",
    }

    for hp_name, hp_fields in heat_pumps.items():
        us_fields: dict[str, Any] = {}

        for field_name, value in hp_fields.items():
            new_name = FIELD_RENAME.get(field_name, field_name)
            us_fields[new_name] = value

        for flow_field, method_field in FLOW_RATE_METHOD_FIELDS.items():
            if flow_field in us_fields:
                us_fields[method_field] = "SupplyAirFlowRate"

        us_fields["control_type"] = "SetPoint"
        us_fields["dehumidification_control_type"] = "None"

        HP_ONLY_FIELDS = {
            "maximum_outdoor_dry_bulb_temperature_for_supplemental_heater_operation",
        }
        for hp_field in HP_ONLY_FIELDS:
            us_fields.pop(hp_field, None)

        unitary_systems[hp_name] = us_fields

    del obj[HP_TYPE]

    for _branch_name, branch in obj.get("Branch", {}).items():
        for component in branch.get("components", []):
            if component.get("component_object_type") == HP_TYPE:
                component["component_object_type"] = US_TYPE

    return obj


def make_unitary_system_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], list[UnitarySystem]]:
    """Discover AirLoopHVAC:UnitarySystem objects and instrument them with
    fan mass flow rate + SAT setpoint actuators.

    The control_type is set to "SetPoint" so that the system actively modulates
    coils to meet the outlet node setpoint. Systems converted from
    UnitaryHeatPump already have this field, but setting it again is harmless.
    """
    return make_unitary_controllable(
        obj,
        epjson_type="AirLoopHVAC:UnitarySystem",
        fan_field="supply_fan_name",
        set_control_type=True,
    )


@dataclass
class HeatingOnlyZone:
    zone: str
    heating_setpoint: ActuatorDescription
    equipment_type: Literal["heating_only"] = "heating_only"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        return [self.heating_setpoint]

    def zones(self) -> list[str]:
        return [self.zone]


Baseboard = HeatingOnlyZone

HEATING_ONLY_EQUIPMENT_TYPES: list[str] = [
    "ZoneHVAC:Baseboard:Convective:Electric",
    "ZoneHVAC:Baseboard:Convective:Water",
    "ZoneHVAC:Baseboard:RadiantConvective:Electric",
    "ZoneHVAC:Baseboard:RadiantConvective:Water",
    "ZoneHVAC:UnitHeater",
    "ZoneHVAC:HighTemperatureRadiant",
]


def make_heating_only_controllable(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[HeatingOnlyZone]]:
    """Find all heating-only zone equipment via EquipmentConnections and
    control them via thermostat heating setpoint schedules.

    Covers baseboards, unit heaters, and high-temperature radiant heaters.
    Equipment availability is pinned always-on; the only actuator exposed
    is the zone thermostat heating setpoint.  Zones with multiple
    heating-only devices get a single actuator (deduplicated by zone).
    """

    obj = deepcopy(obj)
    ont = Ontology.from_object(obj)
    g = ont.rdf

    htg_stl_name = create_temp_stl(
        obj, HEATING_ONLY_SP_C.low, HEATING_ONLY_SP_C.high, name="heating only setpoint stl"
    )

    onoff_stl = create_onoff_availability_stl(obj, name="heating only availability")
    always_on_sched = create_schedule_constant(
        obj, onoff_stl, 1, name="heating only always on"
    )

    equip_query = """
        SELECT ?zone ?equipName ?equipType
        WHERE {
            ?equipConn a "ZoneHVAC:EquipmentConnections" .
            ?equipConn idf:zone_name ?zone .
            ?equipConn idf:zone_conditioning_equipment_list_name ?equipList .

            ?equipList a "ZoneHVAC:EquipmentList" .
            ?equipList idf:equipment ?equipHead .
            ?equipHead rdf:rest*/rdf:first ?equipItem .
            ?equipItem idf:zone_equipment_object_type ?equipType .
            ?equipItem idf:zone_equipment_name ?equipName .
        }
    """

    zones_seen: set[str] = set()
    new_devices: list[HeatingOnlyZone] = []

    for heating_type in HEATING_ONLY_EQUIPMENT_TYPES:
        type_literal = rdflib.Literal(heating_type)
        for row in g.query(equip_query, initBindings={"equipType": type_literal}):
            zone = str(row.zone)
            equip_name = str(row.equipName)

            type_section = obj.get(heating_type, {})
            if equip_name in type_section:
                type_section[equip_name]["availability_schedule_name"] = always_on_sched

            if zone in zones_seen:
                continue
            zones_seen.add(zone)

            thermostat_controls = obj.get("ZoneControl:Thermostat", {})
            tc = None
            for _name, candidate in thermostat_controls.items():
                if candidate.get("zone_or_zonelist_name") == zone:
                    tc = candidate
                    break

            if tc is None:
                raise ValueError(
                    f"No ZoneControl:Thermostat found for heating-only zone {zone}"
                )

            control_type = tc.get("control_1_object_type", "")
            control_name = tc["control_1_name"]

            htg_sched = create_schedule_constant(
                obj, htg_stl_name, 18, name=f"heating only htg setpoint {zone}"
            )

            if control_type == "ThermostatSetpoint:DualSetpoint":
                dsp = obj["ThermostatSetpoint:DualSetpoint"][control_name]
                dsp["heating_setpoint_temperature_schedule_name"] = htg_sched
            elif control_type == "ThermostatSetpoint:SingleHeating":
                sp = obj["ThermostatSetpoint:SingleHeating"][control_name]
                sp["setpoint_temperature_schedule_name"] = htg_sched
            else:
                raise ValueError(
                    f"Unsupported thermostat type {control_type!r} "
                    f"for heating-only zone {zone}"
                )

            htg_actuator = ActuatorDescription(
                component_type="Schedule:Constant",
                control_type="Schedule Value",
                component_name=htg_sched,
                units="[C]",
                lower_bound=HEATING_ONLY_SP_C.low,
                upper_bound=HEATING_ONLY_SP_C.high,
            )

            new_devices.append(
                HeatingOnlyZone(zone=zone, heating_setpoint=htg_actuator)
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
    # Per-loop outdoor-air mixer actuator. Exposes the
    # ``Controller:OutdoorAir × Air Mass Flow Rate`` EMS actuator (kg/s).
    # Required for OfficeMedium so the agent can regulate the mixed-air
    # fraction.
    oa_mass_flow: ActuatorDescription
    equipment_type: Literal["vavsystem"] = "vavsystem"

    def actuator_descriptions(self) -> list[ActuatorDescription]:
        out = [self.supply_temp_setpoint]
        for vav in self.terminals:
            out.append(vav.flow_fraction)
            out.append(vav.heating_setpoint)
            out.append(vav.cooling_setpoint)
        out.append(self.oa_mass_flow)
        return out

    def zones(self) -> list[str]:
        return [vav.zone for vav in self.terminals]


def remove_thermostat_ems_overrides(obj: dict[str, Any]) -> None:
    """Remove EMS optimum-start programs that override thermostat setpoint
    schedules.

    ASHRAE 90.1 OfficeMedium buildings include EMS programs that SET
    CLGSETP_SCH / HTGSETP_SCH actuators at BeginTimestepBeforePredictor.
    These fight any external setpoint control and must be removed.

    Strategy: identify EMS:Actuator entries whose target schedule name
    contains ``CLGSETP_SCH`` or ``HTGSETP_SCH``, then cascade-delete the
    programs, calling-managers, sensors, and internal-variables that
    reference them.
    """
    ems_actuators = obj.get("EnergyManagementSystem:Actuator", {})

    # 1. Find EMS actuator names targeting thermostat setpoint schedules.
    target_actuator_names: set[str] = set()
    for name, act in list(ems_actuators.items()):
        comp_name = act.get("actuated_component_unique_name", "")
        if "CLGSETP_SCH" in comp_name or "HTGSETP_SCH" in comp_name:
            target_actuator_names.add(name)

    if not target_actuator_names:
        return

    # 2. Find EMS programs that SET any of these actuators.
    ems_programs = obj.get("EnergyManagementSystem:Program", {})
    programs_to_remove: set[str] = set()
    for prog_name, prog in ems_programs.items():
        lines = " ".join(l.get("program_line", "") for l in prog.get("lines", []))
        if any(act_name in lines for act_name in target_actuator_names):
            programs_to_remove.add(prog_name)

    # 3. Collect sensor / internal-variable names used by those programs.
    sensor_names: set[str] = set()
    ivar_names: set[str] = set()
    for prog_name in programs_to_remove:
        prog = ems_programs[prog_name]
        lines = " ".join(l.get("program_line", "") for l in prog.get("lines", []))
        for sname in obj.get("EnergyManagementSystem:Sensor", {}):
            if sname in lines:
                sensor_names.add(sname)
        for ivname in obj.get("EnergyManagementSystem:InternalVariable", {}):
            if ivname in lines:
                ivar_names.add(ivname)

    # 4. Delete calling managers that reference removed programs.
    for pcm_name in list(obj.get("EnergyManagementSystem:ProgramCallingManager", {})):
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

    temp_stl_name = create_temp_stl(
        obj, VAV_SUPPLY_TEMP_C.low, VAV_SUPPLY_TEMP_C.high, name="vav supply temp stl"
    )
    htg_stl_name = create_temp_stl(
        obj, VAV_HEATING_SP_C.low, VAV_HEATING_SP_C.high, name="vav heating setpoint stl"
    )
    clg_stl_name = create_temp_stl(
        obj, VAV_COOLING_SP_C.low, VAV_COOLING_SP_C.high, name="vav cooling setpoint stl"
    )
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
            lower_bound=VAV_SUPPLY_TEMP_C.low,
            upper_bound=VAV_SUPPLY_TEMP_C.high,
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
            lower_bound=FLOW_FRACTION.low,
            upper_bound=FLOW_FRACTION.high,
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
            lower_bound=VAV_HEATING_SP_C.low,
            upper_bound=VAV_HEATING_SP_C.high,
        )
        clg_actuator = ActuatorDescription(
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=clg_sched,
            units="[C]",
            lower_bound=VAV_COOLING_SP_C.low,
            upper_bound=VAV_COOLING_SP_C.high,
        )

        return htg_actuator, clg_actuator

    # Pin fan / air-loop availability always-on for RL control.  Without
    # this override, OfficeMedium (and other VAV prototypes) inherit the
    # DOE ``HVACOperationSchd`` which turns supply fans off nights / Sundays
    # and re-enables them only during occupied hours -- that schedule
    # accounts for ~45 % of the year.  Whenever the fan is off, the VAV
    # damper cannot deliver conditioned air regardless of what the agent
    # commands on the SAT or flow-fraction actuators, which mechanically
    # caps cooling control.  ``AvailabilityManager:NightCycle`` on the
    # air loop is also neutralized for the same reason.
    vav_onoff_stl = create_onoff_availability_stl(
        obj, name="vav always on availability"
    )
    vav_always_on_sched = create_schedule_constant(
        obj, vav_onoff_stl, 1, name="vav always on availability"
    )

    def _find_oa_controllers_for_loop(loop_name: str) -> list[str]:
        """Return the ``Controller:OutdoorAir`` names attached to *loop_name*.

        Walks ``AirLoopHVAC`` → ``BranchList`` → ``Branch`` → components,
        keeping only ``AirLoopHVAC:OutdoorAirSystem`` components on the
        loop, then resolves their controller list to the
        ``Controller:OutdoorAir`` entries.
        """
        oa_systems_on_loop: list[str] = []
        query = """
            SELECT ?loop ?compType ?compName
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
                ?comp idf:component_object_type ?compType .
                ?comp idf:component_name ?compName .
            }
        """
        for row in g.query(query):
            if str(row.loop) != loop_name:
                continue
            if str(row.compType) != "AirLoopHVAC:OutdoorAirSystem":
                continue
            oa_systems_on_loop.append(str(row.compName))

        oa_systems = obj.get("AirLoopHVAC:OutdoorAirSystem", {})
        controller_lists = obj.get("AirLoopHVAC:ControllerList", {})
        out: list[str] = []
        for oa_system_name in oa_systems_on_loop:
            oa_system = oa_systems.get(oa_system_name)
            if oa_system is None:
                raise KeyError(
                    f"AirLoopHVAC:OutdoorAirSystem {oa_system_name!r} referenced "
                    f"by loop {loop_name!r} not found in epJSON."
                )
            controller_list_name = oa_system.get("controller_list_name")
            if not controller_list_name:
                raise KeyError(
                    f"AirLoopHVAC:OutdoorAirSystem {oa_system_name!r} has no "
                    f"controller_list_name."
                )
            controller_list = controller_lists.get(controller_list_name)
            if controller_list is None:
                raise KeyError(
                    f"AirLoopHVAC:ControllerList {controller_list_name!r} not "
                    f"found in epJSON."
                )
            for key, value in controller_list.items():
                # entries are named controller_<i>_object_type / controller_<i>_name
                if not key.endswith("_object_type"):
                    continue
                if value != "Controller:OutdoorAir":
                    continue
                name_key = key.replace("_object_type", "_name")
                controller_name = controller_list.get(name_key)
                if not controller_name:
                    raise KeyError(
                        f"AirLoopHVAC:ControllerList {controller_list_name!r} "
                        f"entry {key!r}=Controller:OutdoorAir is missing its "
                        f"{name_key!r}."
                    )
                out.append(str(controller_name))
        return out

    def install_oa_mixer_actuator(
        loop_name: str, controller_name: str
    ) -> ActuatorDescription:
        """Pin the ``Controller:OutdoorAir`` minimum-OA schedule to an
        always-on constant and expose the ``Outdoor Air Controller × Air
        Mass Flow Rate`` EMS actuator (kg/s).

        EnergyPlus ignores EMS overrides on this actuator while the
        ``minimum_outdoor_air_schedule_name`` field constrains the
        minimum flow upwards or while a non-trivial maximum schedule
        clamps it.  We rebind both fields to ``vav_always_on_sched`` so
        the agent's EMS write is authoritative.
        """
        controllers = obj.get("Controller:OutdoorAir", {})
        controller = controllers.get(controller_name)
        if controller is None:
            raise KeyError(
                f"Controller:OutdoorAir {controller_name!r} (loop "
                f"{loop_name!r}) not found in epJSON."
            )
        # The minimum-OA schedule is a fraction (0-1) of the controller's
        # minimum_outdoor_air_flow_rate.  Pinning it to the always-on
        # availability schedule (value 1.0) means E+ will allow the EMS
        # actuator to drive the OA flow without a schedule-imposed floor.
        for sched_field in (
            "minimum_outdoor_air_schedule_name",
            "minimum_fraction_of_outdoor_air_schedule_name",
            "maximum_fraction_of_outdoor_air_schedule_name",
            "time_of_day_economizer_control_schedule_name",
        ):
            if sched_field in controller:
                controller[sched_field] = vav_always_on_sched
        return ActuatorDescription(
            component_type="Outdoor Air Controller",
            control_type="Air Mass Flow Rate",
            component_name=controller_name,
            units="[kg/s]",
            lower_bound=OA_MASS_FLOW_KGS.low,
            upper_bound=OA_MASS_FLOW_KGS.high,
        )

    def _find_supply_fans_for_loop(loop_name: str) -> list[tuple[str, str]]:
        """Return ``[(fan_type, fan_name), ...]`` for the supply branches of
        *loop_name*, by walking ``AirLoopHVAC`` → ``BranchList`` → ``Branch``
        components.  The ontology query is fan-type-agnostic; we filter in
        Python by matching ``component_object_type`` against known fan
        object types.
        """
        fan_types = {
            "Fan:SystemModel",
            "Fan:OnOff",
            "Fan:ConstantVolume",
            "Fan:VariableVolume",
        }
        found: list[tuple[str, str]] = []
        query = """
            SELECT ?loop ?fanType ?fanName
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
                ?comp idf:component_object_type ?fanType .
                ?comp idf:component_name ?fanName .
            }
        """
        for row in g.query(query):
            if str(row.loop) != loop_name:
                continue
            if str(row.fanType) not in fan_types:
                continue
            found.append((str(row.fanType), str(row.fanName)))
        return found

    loops = []
    for loop_name, (supply_outlet, zone_terminals) in loops_dict.items():
        for fan_type, fan_name in _find_supply_fans_for_loop(loop_name):
            _set_fan_always_on(obj, fan_name, vav_always_on_sched)
            logger.info(
                "VAV always-on: overrode availability of %s %r on loop %r",
                fan_type,
                fan_name,
                loop_name,
            )
        _ensure_always_on_availability(obj, loop_name, vav_always_on_sched)

        # Discover the OA mixer attached to this loop.  EnergyPlus VAV
        # systems carry exactly one ``AirLoopHVAC:OutdoorAirSystem`` per
        # supply branch, resolving to one ``Controller:OutdoorAir``.  We
        # do not assume a fixed loop count across the building but
        # we do assume one OA controller per loop -- a VAV loop with
        # zero or multiple OA controllers is malformed for OfficeMedium
        # and we fail loudly.
        oa_controllers = _find_oa_controllers_for_loop(loop_name)
        if len(oa_controllers) != 1:
            raise ValueError(
                f"VAV loop {loop_name!r} resolved to {len(oa_controllers)} "
                f"Controller:OutdoorAir objects (expected 1). "
                f"Controllers found: {oa_controllers!r}."
            )
        oa_act = install_oa_mixer_actuator(loop_name, oa_controllers[0])

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
                oa_mass_flow=oa_act,
            )
        )

    return obj, loops


# The purpose of this type (compared to types.Equipment) is to actually list all
# the different types of things an `Equipment` can be. If we don't do that, the
# cattrs library cant rehydrate the dataclasses correctly.
AnyEquipment = VAVSystem | UnitarySystem | HeatingOnlyZone


def make_all_equipment(
    json_obj: dict[str, Any],
) -> tuple[dict[str, Any], Sequence[AnyEquipment]]:
    json_obj = convert_heat_pumps_to_unitary_systems(json_obj)

    all_functions = [
        make_unitary_system_controllable,
        make_vav_system_controllable,
        make_heating_only_controllable,
    ]
    all_equipment: list[AnyEquipment] = []
    for func in all_functions:
        json_obj, devices = func(json_obj)
        all_equipment += devices

    return json_obj, all_equipment


def make_controllable(
    input_epjson: Realizable,
    *,
    controls: Sequence[str] | None = None,
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
