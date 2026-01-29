from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RolloutPaths:
    out_dir: Path
    csv_path: Path
    npz_path: Path
    config_path: Path
    plot_temperature_path: Path
    plot_energy_path: Path
    plot_actuators_path: Path


def make_rollout_paths(run_dir: Path) -> RolloutPaths:
    run_dir.mkdir(parents=True, exist_ok=True)
    return RolloutPaths(
        out_dir=run_dir,
        csv_path=run_dir / "rollout.csv",
        npz_path=run_dir / "rollout.npz",
        config_path=run_dir / "config_resolved.json",
        plot_temperature_path=run_dir / "temperature.png",
        plot_energy_path=run_dir / "energy.png",
        plot_actuators_path=run_dir / "actuators.png",
    )


def require_env_metadata_list_str(env: Any, key: str) -> list[str]:
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("env.metadata missing")
    raw = env.metadata.get(key)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"env.metadata['{key}'] must be a list[str]")
    return raw


def find_first_zone_air_temp_index(observation_names: list[str]) -> int:
    for i, name in enumerate(observation_names):
        if str(name).lower().startswith("zone air temperature"):
            return int(i)
    for i, name in enumerate(observation_names):
        if "zone air temperature" in str(name).lower():
            return int(i)
    raise RuntimeError("Could not find a 'Zone Air Temperature' entry in observation_names")


def find_controlled_zone_air_temp_index(
    observation_names: list[str],
    *,
    controlled_zones: list[str] | None,
) -> int:
    """
    Prefer a Zone Air Temperature observation that corresponds to a controlled zone.

    This avoids accidentally controlling a temperature from an uncontrolled zone
    (e.g., garage) when the HVAC does not serve that zone.
    """
    if not controlled_zones:
        return find_first_zone_air_temp_index(observation_names)

    controlled = {str(z).strip().lower() for z in controlled_zones if str(z).strip()}
    if not controlled:
        return find_first_zone_air_temp_index(observation_names)

    prefix = "zone air temperature"
    for i, name in enumerate(observation_names):
        s = str(name).strip()
        sl = s.lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if zone_part in controlled:
            return int(i)

    # Fallback: best-effort substring match (for ontologies that decorate zone names).
    for i, name in enumerate(observation_names):
        sl = str(name).strip().lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if any(z in zone_part or zone_part in z for z in controlled):
            return int(i)

    return find_first_zone_air_temp_index(observation_names)


def _find_obs_index_by_name(observation_names: list[str], *, key: str) -> int:
    """
    Find the index of a time-like scalar observation by its canonical key.

    `flat_observation_info` uses canonical slot names like "time_of_day".
    This helper also supports best-effort substring matching for robustness.
    """
    key_l = key.strip().lower()
    for i, name in enumerate(observation_names):
        if str(name).strip().lower() == key_l:
            return int(i)
    for i, name in enumerate(observation_names):
        if key_l in str(name).strip().lower():
            return int(i)
    raise RuntimeError(f"Could not find observation '{key}' in observation_names")


def find_time_of_day_index(observation_names: list[str]) -> int:
    return _find_obs_index_by_name(observation_names, key="time_of_day")


def find_day_of_week_index(observation_names: list[str]) -> int:
    return _find_obs_index_by_name(observation_names, key="day_of_week")


def find_zone_air_temp_index_for_zone(observation_names: list[str], *, zone_name: str) -> int:
    """
    Find the flat observation index for "Zone Air Temperature {zone}".

    Matches the naming conventions used by `flat_observation_info`.
    """
    zn = zone_name.strip().lower()
    prefix = "zone air temperature"

    # Exact-ish match.
    for i, name in enumerate(observation_names):
        s = str(name).strip()
        sl = s.lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if zone_part == zn:
            return int(i)

    # Fallback substring match.
    for i, name in enumerate(observation_names):
        sl = str(name).strip().lower()
        if not sl.startswith(prefix):
            continue
        zone_part = sl[len(prefix) :].strip()
        if zn in zone_part or zone_part in zn:
            return int(i)

    raise RuntimeError(f"Could not find Zone Air Temperature for zone '{zone_name}'")


def find_obs_index_by_exact_name(observation_names: list[str], *, name: str) -> int | None:
    """
    Find an observation index by exact (case-insensitive) match.
    Returns None if not present.
    """
    key = name.strip().lower()
    for i, n in enumerate(observation_names):
        if str(n).strip().lower() == key:
            return int(i)
    return None


def find_action_index(
    action_names: list[str], component_type: str, control_type: str
) -> int | None:
    ct = component_type.strip().lower()
    ctrl = control_type.strip().lower()
    for i, name in enumerate(action_names):
        parts = str(name).split("::")
        if len(parts) < 2:
            continue
        if parts[0].strip().lower() == ct and parts[1].strip().lower() == ctrl:
            return int(i)
    return None


def find_action_indices(
    action_names: list[str],
    *,
    component_type_prefix: str | None = None,
    control_type: str | None = None,
    component_name_contains: str | None = None,
) -> list[int]:
    """
    Find all indices in `action_names` that match simple filters on the triplet
    "{component_type}::{control_type}::{component_name}".
    """
    out: list[int] = []
    ct_prefix = component_type_prefix.strip().lower() if component_type_prefix else None
    ctrl = control_type.strip().lower() if control_type else None
    name_sub = component_name_contains.strip().lower() if component_name_contains else None

    for i, name in enumerate(action_names):
        parts = str(name).split("::")
        if len(parts) < 3:
            continue
        ct = parts[0].strip().lower()
        c = parts[1].strip().lower()
        cn = parts[2].strip().lower()
        if ct_prefix is not None and not ct.startswith(ct_prefix):
            continue
        if ctrl is not None and c != ctrl:
            continue
        if name_sub is not None and name_sub not in cn:
            continue
        out.append(i)
    return out

