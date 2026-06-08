from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


def get_hvac_actuators(edd_path: Path) -> list[dict[str, str]]:
    """Extract HVAC-related actuator names from an EnergyPlus .edd file.

    Searches for actuators related to:
    - Coil speed/stage control (heating/cooling coils and unitary systems)
    - Fan air mass flow rate control
    - AirTerminal mass flow rate controls
    - AirLoopHVAC availability status override (force system on/off)
    - ZoneHVAC equipment actuators (e.g., PTAC, window AC, baseboards, etc.)

    Returns:
        List of dictionaries containing actuator information for get_actuator_handle().
        Each dictionary has keys: 'component_name', 'component_type', 'control_type', 'units'
    """
    hvac_actuators = []

    # Read the .edd file
    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Keywords to identify HVAC actuators (case-insensitive search)
    hvac_keywords = [
        "Coil Speed Control",
        "Fan Air Mass Flow Rate",
        # Air loop availability override (ForceOff / CycleOn / CycleOnZoneFansOnly)
        "AirLoopHVAC,Availability Status",
        # Zone equipment (cooling/heating terminals)
        "ZoneHVAC:",
    ]

    # Lines to exclude (schedules, not direct HVAC equipment controls)
    exclude_keywords = [
        "Schedule:Year",
        "Schedule:File",
        "Schedule:Compact",
        "Schedule:Constant",
        "ElectricEquipment",
        "OtherEquipment",
        "Surface,",
        "Weather Data",
        "Material,",
        "People,",
        "Lights,",
        "Zone,",
        "System Node Setpoint",
        "Plant Component",
        "Autosized",
    ]

    for line in lines:
        line_stripped = line.strip()

        # Skip comments and empty lines
        if not line_stripped or line_stripped.startswith("!"):
            continue

        # Check if line contains HVAC-related keywords
        line_lower = line_stripped.lower()

        # First check if line should be excluded
        should_exclude = any(excl.lower() in line_lower for excl in exclude_keywords)
        if should_exclude:
            continue

        # Check if line contains any HVAC keywords
        is_hvac = any(keyword.lower() in line_lower for keyword in hvac_keywords)

        # Also check for specific component types that are HVAC-related
        if not is_hvac:
            # Additional patterns for HVAC equipment
            if "airterminal:" in line_lower:
                is_hvac = True
            elif "fan," in line_lower and "mass flow" in line_lower:
                is_hvac = True
            elif "coil" in line_lower and (
                "speed" in line_lower or "stage" in line_lower
            ):
                is_hvac = True
            elif "airloophvac," in line_lower and "availability status" in line_lower:
                is_hvac = True
            elif "zonehvac:" in line_lower:
                is_hvac = True

        if is_hvac:
            # Parse the actuator line
            # Format: EnergyManagementSystem:Actuator Available,<Component Name>,<Component Type>,<Control Type>,<Units>
            parts = line_stripped.split(",", maxsplit=4)

            if len(parts) >= 5:
                actuator_dict = {
                    "component_name": parts[1].strip(),
                    "component_type": parts[2].strip(),
                    "control_type": parts[3].strip(),
                    "units": parts[4].strip(),
                }
                hvac_actuators.append(actuator_dict)

    return hvac_actuators


def get_schedule_value_actuators(edd_path: Path) -> list[dict[str, str]]:
    """
    Extract schedule value actuators from an EnergyPlus `.edd` file.

    These actuators are typically of the form:
      EnergyManagementSystem:Actuator Available,<Schedule Name>,Schedule:*,Schedule Value,[ ]
    """
    out: list[dict[str, str]] = []
    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if "energymanagementsystem:actuator available" not in line.lower():
                continue
            if "schedule value" not in line.lower():
                continue

            parts = line.split(",", maxsplit=4)
            if len(parts) < 5:
                continue
            component_type = parts[2].strip()
            if not component_type.lower().startswith("schedule:"):
                continue

            out.append(
                {
                    "component_name": parts[1].strip(),
                    "component_type": component_type,
                    "control_type": parts[3].strip(),
                    "units": parts[4].strip(),
                }
            )
    return out


def get_b2b_scheduled_node_setpoint_actuators(
    edd_path: Path,
    *,
    schedule_names: Sequence[str],
) -> list[dict[str, str]]:
    """
    Return schedule-value actuators for the schedules used by `B2B Node Temp SPM ...`.

    These correspond to lines like:
      EnergyManagementSystem:Actuator Available,<Schedule Name>,Schedule:Constant,Schedule Value,[ ]
    """
    # EnergyPlus often uppercases object names in `.edd`, so match case-insensitively.
    want_upper = {str(s).strip().upper() for s in schedule_names if str(s).strip()}
    if not want_upper:
        return []

    out: list[dict[str, str]] = []
    for a in iter_edd_actuators(edd_path):
        ct = a.component_type.strip().lower()
        ctrl = a.control_type.strip().lower()
        if not ct.startswith("schedule:"):
            continue
        if ctrl != "schedule value":
            continue
        if a.component_name.strip().upper() not in want_upper:
            continue
        out.append(a.to_dict())

    return out


def get_zone_temperature_control_actuators(edd_path: Path) -> list[dict[str, str]]:
    """
    Extract Zone Temperature Control actuators (EMS overrides of thermostat setpoints).

    This is useful when we want to "force" EnergyPlus into heating/cooling mode without
    relying on the building's thermostat schedules.
    """
    out: list[dict[str, str]] = []

    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if "zone temperature control" not in line.lower():
                continue
            # Typical `.edd` line format:
            # EnergyManagementSystem:Actuator Available,<Component Name>,<Component Type>,<Control Type>,<Units>
            parts = line.split(",", maxsplit=4)
            if len(parts) < 5:
                continue
            control_type = parts[3].strip()
            if control_type not in ("Heating Setpoint", "Cooling Setpoint"):
                continue
            out.append(
                {
                    "component_name": parts[1].strip(),
                    "component_type": parts[2].strip(),
                    "control_type": control_type,
                    "units": parts[4].strip(),
                }
            )

    return out


@dataclass(frozen=True, slots=True)
class EddActuatorDescriptor:
    """
    Strongly-typed representation of a single EMS actuator availability dictionary entry.

    This corresponds to `.edd` lines such as:
      EnergyManagementSystem:Actuator Available,<Component Name>,<Component Type>,<Control Type>,<Units>
    """

    component_name: str
    component_type: str
    control_type: str
    units: str

    @classmethod
    def from_edd_line(cls, line: str) -> "EddActuatorDescriptor | None":
        s = str(line).strip()
        if not s or s.startswith("!"):
            return None
        if "energymanagementsystem:actuator available" not in s.lower():
            return None

        parts = s.split(",", maxsplit=4)
        if len(parts) < 5:
            return None
        component_name = parts[1].strip()
        component_type = parts[2].strip()
        control_type = parts[3].strip()
        units = parts[4].strip()
        if not component_name or not component_type or not control_type:
            return None
        return cls(
            component_name=component_name,
            component_type=component_type,
            control_type=control_type,
            units=units,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "component_name": self.component_name,
            "component_type": self.component_type,
            "control_type": self.control_type,
            "units": self.units,
        }


def iter_edd_actuators(edd_path: Path) -> Iterator[EddActuatorDescriptor]:
    """
    Yield actuator descriptors from an EnergyPlus `.edd` actuator availability dictionary.
    """
    with open(edd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            desc = EddActuatorDescriptor.from_edd_line(raw)
            if desc is not None:
                yield desc


def get_airflow_and_coil_node_setpoint_actuators(
    edd_path: Path,
    *,
    unitary_outlet_nodes: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """
    Extract actuators needed to control zone temperature via:

    - Fan air mass flow rate:
        <Fan Name>, Fan, Fan Air Mass Flow Rate, [kg/s]
    - Coil control via system node setpoints (Temperature Setpoint):
        <HEATING COIL NODE>, System Node Setpoint, Temperature Setpoint, [C]
        <SUPPLEMENTAL COIL NODE>, System Node Setpoint, Temperature Setpoint, [C]
        <COOLING COIL NODE>, System Node Setpoint, Temperature Setpoint, [C]
    - System availability override:
        <AirLoop Name>, AirLoopHVAC, Availability Status, [ ]

    This function is intentionally targeted (vs returning every System Node Setpoint),
    so action spaces stay compact and stable across buildings.
    """
    desired_node_suffixes = (
        "HEATING COIL NODE",
        "SUPPLEMENTAL COIL NODE",
        "COOLING COIL NODE",
    )

    def matches_any_suffix(node_name: str) -> bool:
        up = node_name.strip().upper()
        return any(sfx in up for sfx in desired_node_suffixes)

    outlet_nodes_norm: set[str] = set()
    if unitary_outlet_nodes is not None:
        for n in unitary_outlet_nodes:
            if isinstance(n, str) and n.strip():
                outlet_nodes_norm.add(n.strip().upper())

    selected: list[EddActuatorDescriptor] = []
    for a in iter_edd_actuators(edd_path):
        ct = a.component_type.strip().lower()
        ctrl = a.control_type.strip().lower()

        # 1) Fan air mass flow rate
        if ct == "fan" and ctrl == "fan air mass flow rate":
            selected.append(a)
            continue

        # 2) Coil node temperature setpoint actuators
        if ct == "system node setpoint" and ctrl == "temperature setpoint":
            if matches_any_suffix(a.component_name) or (
                outlet_nodes_norm
                and a.component_name.strip().upper() in outlet_nodes_norm
            ):
                selected.append(a)
            continue

        # 3) AirLoop availability override (force system available)
        if ct == "airloophvac" and ctrl == "availability status":
            selected.append(a)
            continue

    # Stable ordering + dedup
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for a in selected:
        key = f"{a.component_type}::{a.control_type}::{a.component_name}"
        if key in seen:
            continue
        seen.add(key)
        out.append(a.to_dict())
    return out
