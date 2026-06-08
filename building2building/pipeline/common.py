import contextlib
import os
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def chdir(path: Path):
    """
    Context manager to temporarily change the current working directory.
    Reimplementation of contextlib.chdir for Python < 3.11.
    """
    old_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


def _ensure_unique_object_name(existing: dict[str, Any], base: str) -> str:
    """
    Generate a unique key for an epJSON object map (e.g. Schedule:Constant),
    given a desired base name.
    """
    name = base
    if name not in existing:
        return name
    suffix = 2
    while f"{base} {suffix}" in existing:
        suffix += 1
    return f"{base} {suffix}"


def _resolve_onoff_schedule_type_limits_name(epjson: dict[str, Any]) -> str:
    """
    Return a ScheduleTypeLimits name suitable for a 0/1 availability schedule.
    Creates one if none exist.
    """
    stl = epjson.setdefault("ScheduleTypeLimits", {})
    if not isinstance(stl, dict):
        raise TypeError("epjson['ScheduleTypeLimits'] must be a dict if present")

    for preferred in ("OnOff", "OnOff 1", "On-Off", "On/Off"):
        if preferred in stl:
            return preferred

    for k in stl.keys():
        if "onoff" in str(k).replace(" ", "").lower():
            return str(k)

    if stl:
        return str(next(iter(stl.keys())))

    # Create a minimal discrete availability type limit.
    name = "B2B OnOff"
    stl[name] = {
        "lower_limit_value": 0,
        "upper_limit_value": 1,
        "numeric_type": "Discrete",
        "unit_type": "Availability",
    }
    return name


def _resolve_temperature_schedule_type_limits_name(epjson: dict[str, Any]) -> str:
    """
    Return a ScheduleTypeLimits name suitable for a temperature schedule.
    Creates one if none exist.
    """
    stl = epjson.setdefault("ScheduleTypeLimits", {})
    if not isinstance(stl, dict):
        raise TypeError("epjson['ScheduleTypeLimits'] must be a dict if present")

    for preferred in ("Temperature", "Temperature 1"):
        if preferred in stl:
            return preferred

    for k in stl.keys():
        if "temperature" in str(k).replace(" ", "").lower():
            return str(k)

    # Create a minimal temperature type limit.
    name = "B2B Temperature"
    stl[name] = {
        "lower_limit_value": -100,
        "upper_limit_value": 200,
        "numeric_type": "Continuous",
        "unit_type": "Temperature",
    }
    return name
