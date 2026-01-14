import json
from pathlib import Path
from typing import Any

from building2building.pipeline.common import (
    _ensure_unique_object_name,
    _resolve_onoff_schedule_type_limits_name,
)
from building2building.store import OUTPUT, derivation


@derivation("baseboard-availability-control")
def AddBaseboardAvailabilityControl(input: Path) -> Path:
    """
    If the model contains ZoneHVAC:Baseboard:Convective:Electric, make each baseboard's
    Availability Schedule uniquely controllable via EMS by assigning it a dedicated
    Schedule:Constant (default=1).
    """
    dst = OUTPUT.get()

    with open(input, "r", encoding="utf-8") as f:
        epjson: dict[str, Any] = json.load(f)

    baseboards = epjson.get("ZoneHVAC:Baseboard:Convective:Electric", {})
    if not isinstance(baseboards, dict) or not baseboards:
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(epjson, f, indent=4)
        return dst

    stl_name = _resolve_onoff_schedule_type_limits_name(epjson)
    sched_const = epjson.setdefault("Schedule:Constant", {})
    if not isinstance(sched_const, dict):
        raise TypeError("epjson['Schedule:Constant'] must be a dict if present")

    for bb_name, bb_obj_any in baseboards.items():
        if not isinstance(bb_obj_any, dict):
            continue

        base_sched_name = f"B2B Baseboard Availability {bb_name}"
        sched_name = _ensure_unique_object_name(sched_const, base_sched_name)

        if sched_name not in sched_const:
            sched_const[sched_name] = {
                "hourly_value": 1,
                "schedule_type_limits_name": stl_name,
            }

        bb_obj_any["availability_schedule_name"] = sched_name

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(epjson, f, indent=4)
    return dst


def get_baseboard_availability_schedule_names(epjson: dict[str, Any]) -> list[str]:
    """
    Return availability schedule names used by ZoneHVAC:Baseboard:Convective:Electric objects.
    """
    baseboards = epjson.get("ZoneHVAC:Baseboard:Convective:Electric", {})
    if not isinstance(baseboards, dict):
        return []
    names: list[str] = []
    for _bb_name, bb_obj_any in baseboards.items():
        if not isinstance(bb_obj_any, dict):
            continue
        sched = bb_obj_any.get("availability_schedule_name")
        if isinstance(sched, str) and sched.strip():
            names.append(sched.strip())
    return names

