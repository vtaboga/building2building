"""Observation and action name parsing helpers.

These functions inspect ``env.metadata["observation_names"]`` and
``env.metadata["action_names"]`` to locate specific slots by name,
enabling controllers and evaluation code to work with any B2B
environment without hard-coding indices.
"""

from __future__ import annotations


def find_obs_index(observation_names: list[str], key: str) -> int:
    """Find an observation slot by exact case-insensitive name match."""
    key_l = key.strip().lower()
    for i, name in enumerate(observation_names):
        if name.strip().lower() == key_l:
            return i
    raise RuntimeError(f"Could not find observation {key!r} in observation_names")


def find_obs_index_optional(observation_names: list[str], key: str) -> int | None:
    """Like :func:`find_obs_index` but returns ``None`` when missing."""
    key_l = key.strip().lower()
    for i, name in enumerate(observation_names):
        if name.strip().lower() == key_l:
            return i
    return None


def find_zone_air_temp_index(observation_names: list[str], zone_name: str) -> int:
    """Find the observation index for a zone's air temperature."""
    prefix = "zone air temperature"
    zn = zone_name.strip().lower()
    for i, name in enumerate(observation_names):
        sl = name.strip().lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if zone_part == zn:
            return i
    for i, name in enumerate(observation_names):
        sl = name.strip().lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if zn in zone_part or zone_part in zn:
            return i
    raise RuntimeError(f"Could not find Zone Air Temperature for zone {zone_name!r}")


def find_first_zone_air_temp_index(observation_names: list[str]) -> int:
    """Find the first zone air temperature observation."""
    for i, name in enumerate(observation_names):
        if name.strip().lower().startswith("zone air temperature"):
            return i
    raise RuntimeError("No Zone Air Temperature entry found")


def find_action_index(
    action_names: list[str],
    component_type: str,
    control_type: str,
) -> int | None:
    """Find an action index matching component_type::control_type."""
    ct = component_type.strip().lower()
    ctrl = control_type.strip().lower()
    for i, name in enumerate(action_names):
        parts = name.split("::")
        if len(parts) < 2:
            continue
        if parts[0].strip().lower() == ct and parts[1].strip().lower() == ctrl:
            return i
    return None


def find_action_indices(
    action_names: list[str],
    *,
    component_type_prefix: str | None = None,
    control_type: str | None = None,
    component_name_contains: str | None = None,
) -> list[int]:
    """Find all action indices matching the given filters.

    Action names follow the format
    ``"{component_type}::{control_type}::{component_name}"``.
    """
    out: list[int] = []
    ct_pfx = component_type_prefix.strip().lower() if component_type_prefix else None
    ctrl = control_type.strip().lower() if control_type else None
    name_sub = (
        component_name_contains.strip().lower() if component_name_contains else None
    )

    for i, name in enumerate(action_names):
        parts = name.split("::")
        if len(parts) < 3:
            continue
        ct = parts[0].strip().lower()
        c = parts[1].strip().lower()
        cn = parts[2].strip().lower()
        if ct_pfx is not None and not ct.startswith(ct_pfx):
            continue
        if ctrl is not None and c != ctrl:
            continue
        if name_sub is not None and name_sub not in cn:
            continue
        out.append(i)
    return out
