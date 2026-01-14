import json
import logging
from pathlib import Path
from typing import Any, Sequence

from building2building.pipeline.common import (
    _ensure_unique_object_name,
    _resolve_onoff_schedule_type_limits_name,
    _resolve_temperature_schedule_type_limits_name,
)
from building2building.store import Derivation, OUTPUT, derivation

logger = logging.getLogger(__name__)


def _gather_unitary_and_coil_outlet_nodes(epjson: dict[str, Any]) -> list[str]:
    """
    Collect node names that are useful for verifying setpoint-based unitary control.

    We target:
    - `AirLoopHVAC:UnitarySystem.air_outlet_node_name`
    - The *outlet* node of referenced cooling/heating/supplemental coils (when present).
    """

    def _read_str(obj: Any, key: str) -> str | None:
        if not isinstance(obj, dict):
            return None
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    def _lookup_object(
        epjson0: dict[str, Any], obj_type: str, obj_name: str
    ) -> dict[str, Any] | None:
        table = epjson0.get(obj_type)
        if not isinstance(table, dict):
            return None
        obj = table.get(obj_name)
        if not isinstance(obj, dict):
            return None
        return obj

    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return []

    nodes: list[str] = []

    for _sys_name, sys_any in unitary_any.items():
        if not isinstance(sys_any, dict):
            continue

        # System outlet node (supply air leaving the unitary system).
        if n := _read_str(sys_any, "air_outlet_node_name"):
            nodes.append(n)

        # Coil outlet nodes, if accessible from referenced objects.
        for prefix in ("cooling", "heating", "supplemental_heating"):
            obj_type = _read_str(sys_any, f"{prefix}_coil_object_type")
            obj_name = _read_str(sys_any, f"{prefix}_coil_name")
            if not obj_type or not obj_name:
                continue

            obj = _lookup_object(epjson, obj_type, obj_name)
            if obj is None:
                continue

            # Different coil objects use slightly different naming conventions.
            for outlet_key in ("air_outlet_node_name", "outlet_node_name"):
                if n := _read_str(obj, outlet_key):
                    nodes.append(n)
                    break

    # Stable ordering + dedup
    out: list[str] = []
    seen: set[str] = set()
    for n in nodes:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _gather_unitary_supply_fans(epjson: dict[str, Any]) -> list[str]:
    """Collect `AirLoopHVAC:UnitarySystem.supply_fan_name` values (deduped)."""
    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return []

    fans: list[str] = []
    for _sys_name, sys_any in unitary_any.items():
        if not isinstance(sys_any, dict):
            continue
        fan = sys_any.get("supply_fan_name")
        if isinstance(fan, str) and fan.strip():
            fans.append(fan.strip())

    out: list[str] = []
    seen: set[str] = set()
    for f in fans:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def get_unitary_air_outlet_node_names(epjson: dict[str, Any]) -> list[str]:
    """Return `AirLoopHVAC:UnitarySystem.air_outlet_node_name` values (deduped)."""
    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return []

    nodes: list[str] = []
    for _sys_name, sys_any in unitary_any.items():
        if not isinstance(sys_any, dict):
            continue
        n = sys_any.get("air_outlet_node_name")
        if isinstance(n, str) and n.strip():
            nodes.append(n.strip())

    out: list[str] = []
    seen: set[str] = set()
    for n in nodes:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def add_node_setpoint_diagnostics_inplace(
    epjson: dict[str, Any],
    *,
    reporting_frequency: str = "Timestep",
) -> None:
    """
    Add `Output:Variable` requests that let you verify:
    - what **node temperature setpoints** are being used
    - what the **actual node temperatures** are

    After running EnergyPlus, these appear in `eplusout.csv` / `eplusout.sql` as
    time series columns.
    """
    nodes = _gather_unitary_and_coil_outlet_nodes(epjson)
    fans = _gather_unitary_supply_fans(epjson)
    if not nodes and not fans:
        return

    outvars = epjson.setdefault("Output:Variable", {})
    if not isinstance(outvars, dict):
        raise TypeError("epjson['Output:Variable'] must be a dict if present")

    def _has(var_name: str, key_value: str) -> bool:
        for _k, obj_any in outvars.items():
            if not isinstance(obj_any, dict):
                continue
            if obj_any.get("variable_name") != var_name:
                continue
            if obj_any.get("key_value") != key_value:
                continue
            if obj_any.get("reporting_frequency") != reporting_frequency:
                continue
            return True
        return False

    desired_node_vars = (
        "System Node Setpoint Temperature",
        "System Node Temperature",
        "System Node Mass Flow Rate",
    )

    desired_fan_vars = ("Fan Air Mass Flow Rate",)

    for node in nodes:
        for var_name in desired_node_vars:
            if _has(var_name, node):
                continue
            base = f"Output:Variable {var_name} {node}"
            key = _ensure_unique_object_name(outvars, base)
            outvars[key] = {
                "key_value": node,
                "variable_name": var_name,
                "reporting_frequency": reporting_frequency,
            }

    for fan in fans:
        for var_name in desired_fan_vars:
            if _has(var_name, fan):
                continue
            base = f"Output:Variable {var_name} {fan}"
            key = _ensure_unique_object_name(outvars, base)
            outvars[key] = {
                "key_value": fan,
                "variable_name": var_name,
                "reporting_frequency": reporting_frequency,
            }


def get_b2b_scheduled_setpoint_schedule_names(epjson: dict[str, Any]) -> list[str]:
    """
    Return the `schedule_name` values used by our `B2B Node Temp SPM ...` setpoint managers.

    These schedule names correspond to `Schedule:Constant` objects that can be controlled
    via EMS `Schedule Value` actuators found in `.edd`.
    """
    spm = epjson.get("SetpointManager:Scheduled", {})
    if not isinstance(spm, dict):
        return []
    out: list[str] = []
    for k, obj_any in spm.items():
        if not isinstance(k, str) or not k.startswith("B2B Node Temp SPM "):
            continue
        if not isinstance(obj_any, dict):
            continue
        sched = obj_any.get("schedule_name")
        if isinstance(sched, str) and sched.strip():
            out.append(sched.strip())
    # stable dedup
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


@derivation("node-setpoint-diagnostics")
def AddNodeSetpointDiagnostics(input: Path) -> None:
    dst = OUTPUT.get()
    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    add_node_setpoint_diagnostics_inplace(epjson, reporting_frequency="Timestep")

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)


def add_node_setpoint_diagnostics(epjson_in: Derivation) -> Derivation:
    """Add `Output:Variable` requests for node setpoint temperature diagnostics."""
    return AddNodeSetpointDiagnostics(epjson_in)


def set_unitary_systems_to_setpoint_control_inplace(epjson: dict[str, Any]) -> None:
    """
    In-place edit: set all `AirLoopHVAC:UnitarySystem` objects to SetPoint control.

    Rationale:
    - Many residential mini-split models use `AirLoopHVAC:UnitarySystem.control_type = Load`
      with a controlling zone thermostat. In that mode, changing coil node setpoints and fan
      flow rate often has little effect if the thermostat is not calling for conditioning.
    - For our control mode (fan airflow + coil/node temperature setpoints), we want the unitary
      system to track a supply air temperature setpoint instead of thermostat load.
    """
    unitary_any = epjson.get("AirLoopHVAC:UnitarySystem", {})
    if not isinstance(unitary_any, dict) or not unitary_any:
        return

    # Ensure we have a valid fan operating mode schedule for unitary systems.
    # EnergyPlus expects values in (0, 1] for AirLoopHVAC:UnitarySystem
    # Supply Air Fan Operating Mode Schedule Name (0 can trigger severe errors).
    stl_name = _resolve_onoff_schedule_type_limits_name(epjson)
    sched_const = epjson.setdefault("Schedule:Constant", {})
    if not isinstance(sched_const, dict):
        raise TypeError("epjson['Schedule:Constant'] must be a dict if present")

    fan_mode_schedule_base = "B2B Unitary Fan Operating Mode"
    fan_mode_schedule = _ensure_unique_object_name(sched_const, fan_mode_schedule_base)
    if fan_mode_schedule not in sched_const:
        # Use 1.0 (continuous fan) to satisfy strict validation.
        sched_const[fan_mode_schedule] = {
            "hourly_value": 1,
            "schedule_type_limits_name": stl_name,
        }

    for _name, obj_any in unitary_any.items():
        if not isinstance(obj_any, dict):
            continue
        obj_any["control_type"] = "SetPoint"
        obj_any["supply_air_fan_operating_mode_schedule_name"] = fan_mode_schedule


@derivation("unitary-setpoint-control")
def SetUnitarySystemsToSetpointControl(input: Path) -> None:
    dst = OUTPUT.get()
    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    set_unitary_systems_to_setpoint_control_inplace(epjson)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)


def set_unitary_systems_to_setpoint_control(epjson_in: Derivation) -> Derivation:
    return SetUnitarySystemsToSetpointControl(epjson_in)


def _node_setpoint_category(node_name: str) -> str:
    up = str(node_name).strip().upper()
    if "COOLING COIL NODE" in up:
        return "cooling_coil"
    if "HEATING COIL" in up and "SUPPLEMENTAL" not in up:
        return "heating_coil"
    if "SUPPLEMENTAL COIL" in up:
        return "supplemental_coil"
    return "unitary_outlet"


def ensure_scheduled_node_temperature_setpoints_inplace(
    epjson: dict[str, Any],
    *,
    temperature_c: float = 22.0,
) -> None:
    """
    Ensure the nodes we care about are controlled by `SetpointManager:Scheduled` objects
    whose `schedule_name` is a `Schedule:Constant`.

    Why this is more robust than directly actuating `System Node Setpoint`:
    - Many models already have SetpointManagers on these nodes. If we directly actuate
      the node setpoint, a SetpointManager may overwrite our value later in the timestep.
    - If we instead actuate the *schedule value* used by the SetpointManager, we become
      the authoritative source of the node setpoint EnergyPlus uses and reports.

    This step covers:
    - Unitary system outlet nodes (`AirLoopHVAC:UnitarySystem.air_outlet_node_name`)
    - Coil outlet nodes referenced by the unitary system (when discoverable from epJSON)
    """
    nodes = _gather_unitary_and_coil_outlet_nodes(epjson)
    if not nodes:
        return

    stl_name = _resolve_temperature_schedule_type_limits_name(epjson)
    sched_const = epjson.setdefault("Schedule:Constant", {})
    if not isinstance(sched_const, dict):
        raise TypeError("epjson['Schedule:Constant'] must be a dict if present")

    spm = epjson.setdefault("SetpointManager:Scheduled", {})
    if not isinstance(spm, dict):
        raise TypeError("epjson['SetpointManager:Scheduled'] must be a dict if present")

    # Track nodes already covered by any scheduled setpoint manager.
    covered_nodes: set[str] = set()
    for _k, obj_any in spm.items():
        if not isinstance(obj_any, dict):
            continue
        node = obj_any.get("setpoint_node_or_nodelist_name")
        if isinstance(node, str) and node.strip():
            covered_nodes.add(node.strip())

    for node in nodes:
        if node in covered_nodes:
            continue

        cat = _node_setpoint_category(node)
        schedule_base = f"B2B Node Temp SP {cat} {node}"
        schedule_name = _ensure_unique_object_name(sched_const, schedule_base)
        if schedule_name not in sched_const:
            sched_const[schedule_name] = {
                "hourly_value": float(temperature_c),
                "schedule_type_limits_name": stl_name,
            }

        spm_base = f"B2B Node Temp SPM {cat} {node}"
        spm_name = _ensure_unique_object_name(spm, spm_base)
        spm[spm_name] = {
            "control_variable": "Temperature",
            "schedule_name": schedule_name,
            "setpoint_node_or_nodelist_name": node,
        }
        covered_nodes.add(node)


@derivation("scheduled-node-temp-setpoints")
def EnsureScheduledNodeTemperatureSetpoints(
    input: Path,
    temperature_c: float = 22.0,
) -> None:
    dst = OUTPUT.get()
    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    ensure_scheduled_node_temperature_setpoints_inplace(
        epjson, temperature_c=temperature_c
    )

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)


def ensure_scheduled_node_temperature_setpoints(
    epjson_in: Derivation, *, temperature_c: float = 22.0
) -> Derivation:
    return EnsureScheduledNodeTemperatureSetpoints(epjson_in, temperature_c=temperature_c)

