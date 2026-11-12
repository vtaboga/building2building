"""Role-labelled channels for a Building2Building environment.

A *channel* is one observation or one actuator of the environment, described
with a stable key, a name, engineering units, a role and the zone it belongs
to. :class:`ChannelMap` reads them from ``env.metadata`` and converts between
the environment's flat observation/action vectors and dictionaries keyed by
channel key, which is what :class:`~building2building.bacnet.points.PointStore`
consumes.

Supported equipment: unitary systems (one unit per zone, exposing a supply air
temperature setpoint and a fan air mass flow) and heating-only zone equipment
(exposing a heating setpoint), which covers ``OfficeSmall``,
``RestaurantFastFood``, ``RetailStandalone``, ``Warehouse`` and
``SingleFamilyHouse``. VAV systems (``OfficeMedium``) are not mapped yet. Any
unmapped observation or actuator raises :class:`ValueError` rather than being
silently dropped.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

# (equipment_type, component_type, control_type, units) -> (role, BACnet units)
_ACTUATOR_ROLES = {
    ("unitarysystem", "Fan", "Fan Air Mass Flow Rate", "[kg/s]"): (
        "airflow_command",
        "kg/s",
    ),
    ("unitarysystem", "Schedule:Constant", "Schedule Value", "Temperature"): (
        "sat_command",
        "degC",
    ),
    ("heating_only", "Schedule:Constant", "Schedule Value", "[C]"): (
        "heating_setpoint_command",
        "degC",
    ),
}
_EQUIPMENT_TYPES = frozenset(key[0] for key in _ACTUATOR_ROLES)
# Equipment whose actuators make_env may hold fixed instead of exposing.
_OPTIONAL_EQUIPMENT_TYPES = frozenset({"heating_only"})

_PER_ZONE_OBSERVATIONS = {
    "ZONE AIR TEMPERATURE ": ("zone_temperature", "degC"),
    "zone_occupancy ": ("occupancy", "people"),
    "target_temperature ": ("target_temperature", "degC"),
}

_BUILDING_OBSERVATIONS = {
    "outdoor_temperature": ("outdoor_temperature", "degC"),
    "energy_electricity": ("electricity_kwh", "kWh"),
    "energy_gas": ("gas_kwh", "kWh"),
    "time_of_day": ("time_of_day", "hour"),
    "day_of_week": ("day_of_week", "day"),
    "day_of_year": ("day_of_year", "day"),
}

# Observations that exist in some templates but have no channel.
_SKIPPED_OBSERVATIONS = frozenset({"outdoor_humidity"})

_ENERGY_OBSERVATIONS = frozenset({"energy_electricity", "energy_gas"})


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _has_rescaled_actions(env: gym.Env) -> bool:
    while isinstance(env, gym.Wrapper):
        if isinstance(env, gym.wrappers.RescaleAction):
            return True
        env = env.env
    return False


def command_key(action_name: str) -> str:
    """Return the stable channel key of an actuator from its B2B action name."""
    return "act-" + hashlib.sha256(action_name.encode("utf-8")).hexdigest()[:16]


class ChannelMap:
    """Channels of one environment, and conversions to and from its vectors.

    Args:
        env: An environment from :func:`building2building.make_env`, created
            with ``rescale_action=False`` (the default) so that actions are
            in engineering units.

    Attributes:
        channels: One mapping per channel with ``key``, ``name``, ``units``,
            ``role`` and ``zone`` (``None`` for building-level channels).
            Command channels also have ``writable=True``, ``low``, ``high``
            and ``default`` (the middle of the actuator's range).
        command_channels: The command channels, in the environment's action
            order.

    Raises:
        ValueError: The environment rescales actions, or exposes an
            observation or actuator this layer does not map.
    """

    def __init__(self, env: gym.Env) -> None:
        if _has_rescaled_actions(env):
            raise ValueError(
                "the BACnet layer needs actions in engineering units; "
                "create the environment with rescale_action=False"
            )
        meta = getattr(env, "metadata", None)
        if not isinstance(meta, Mapping):
            raise ValueError("environment did not provide metadata")
        observation_names = meta.get("observation_names")
        action_names = meta.get("action_names")
        if not isinstance(observation_names, (list, tuple)) or not isinstance(
            action_names, (list, tuple)
        ):
            raise ValueError("metadata must include observation_names and action_names")
        self.area = _finite(meta.get("area"), "building area")
        if self.area <= 0:
            raise ValueError("building area must be positive")
        self.observation_names = [str(n) for n in observation_names]
        self.action_names = [str(n) for n in action_names]
        self.building_id = meta.get("building_source_metadata", {}).get("building_id")

        self.channels: list[dict[str, Any]] = []
        self._keys: set[str] = set()
        controlled_zones = {str(z) for z in meta.get("controlled_zones", [])}
        self._add_observations(controlled_zones)
        self._check_zone_coverage(controlled_zones)
        self.command_channels = self._add_commands(meta.get("hvac_equipment", []))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _add(
        self,
        key: str,
        name: str,
        units: str,
        role: str,
        zone: str | None,
        **command: float,
    ) -> dict[str, Any]:
        if key in self._keys:
            raise ValueError(f"duplicate channel {key!r}")
        self._keys.add(key)
        channel: dict[str, Any] = {
            "key": key,
            "name": name,
            "units": units,
            "role": role,
            "zone": zone,
            "writable": bool(command),
        }
        channel.update(command)
        self.channels.append(channel)
        return channel

    def _add_observations(self, controlled_zones: set[str]) -> None:
        for name in self.observation_names:
            if name in _SKIPPED_OBSERVATIONS:
                continue
            if name in _BUILDING_OBSERVATIONS:
                role, units = _BUILDING_OBSERVATIONS[name]
                self._add(name, name, units, role, None)
                continue
            for prefix, (role, units) in _PER_ZONE_OBSERVATIONS.items():
                if name.startswith(prefix):
                    zone = name[len(prefix) :]
                    if role == "zone_temperature" and zone not in controlled_zones:
                        role = "uncontrolled_temperature"
                    self._add(name, name, units, role, zone)
                    break
            else:
                raise ValueError(f"unmapped observation {name!r}")

    def _check_zone_coverage(self, controlled_zones: set[str]) -> None:
        roles_by_zone: dict[str, set[str]] = {}
        for channel in self.channels:
            if channel["zone"] is not None:
                roles_by_zone.setdefault(channel["zone"], set()).add(channel["role"])
        needed = {"zone_temperature", "occupancy", "target_temperature"}
        for zone in sorted(controlled_zones):
            if not needed <= roles_by_zone.get(zone, set()):
                raise ValueError(
                    f"controlled zone {zone!r} is missing temperature, occupancy, "
                    "or target observation"
                )

    def _add_commands(self, equipment: Any) -> list[dict[str, Any]]:
        expected: dict[str, tuple[str, str, str, float, float]] = {}
        optional: set[str] = set()
        for item in equipment:
            equipment_type = getattr(item, "equipment_type", None)
            if equipment_type not in _EQUIPMENT_TYPES:
                continue
            zones = item.zones()
            if len(zones) != 1:
                raise ValueError(f"equipment {item!r} must serve exactly one zone")
            zone = str(zones[0])
            for actuator in item.actuator_descriptions():
                ident = (
                    equipment_type,
                    actuator.component_type,
                    actuator.control_type,
                    actuator.units,
                )
                label = "::".join(ident[1:3] + (actuator.component_name,))
                if ident not in _ACTUATOR_ROLES:
                    raise ValueError(
                        f"unsupported {equipment_type} actuator {label} {actuator.units}"
                    )
                low = _finite(actuator.lower_bound, label)
                high = _finite(actuator.upper_bound, label)
                if low >= high:
                    raise ValueError(f"actuator {label!r} has invalid bounds")
                if label in expected:
                    raise ValueError(f"duplicate actuator {label!r}")
                role, units = _ACTUATOR_ROLES[ident]
                expected[label] = (role, units, zone, low, high)
                if equipment_type in _OPTIONAL_EQUIPMENT_TYPES:
                    optional.add(label)
        unknown = sorted(set(self.action_names) - set(expected))
        hidden = sorted(set(expected) - set(self.action_names) - optional)
        if unknown or hidden or len(set(self.action_names)) != len(self.action_names):
            raise ValueError(
                f"unsupported or unmapped actions; unknown={unknown}, not_exposed={hidden}"
            )
        commands = []
        for name in self.action_names:
            role, units, zone, low, high = expected[name]
            commands.append(
                self._add(
                    command_key(name),
                    name,
                    units,
                    role,
                    zone,
                    low=low,
                    high=high,
                    default=(low + high) / 2,
                )
            )
        return commands

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def defaults(self) -> dict[str, float]:
        """Return each command's default value, keyed by channel key."""
        return {c["key"]: c["default"] for c in self.command_channels}

    def snapshot(self, observation: Any, *, reset: bool = False) -> dict[str, float]:
        """Convert a flat observation into sensor values keyed by channel key.

        Energy observations (Wh/m² over the step) become kWh over the step.
        The observation returned by ``reset()`` closes no interval, so with
        ``reset=True`` energy is reported as zero.
        """
        values = list(observation)
        if len(values) != len(self.observation_names):
            raise ValueError("observation length does not match observation_names")
        snapshot: dict[str, float] = {}
        for name, value in zip(self.observation_names, values):
            if name in _SKIPPED_OBSERVATIONS:
                continue
            number = _finite(value, f"observation {name}")
            if name in _ENERGY_OBSERVATIONS:
                number = 0.0 if reset else number * self.area / 1000.0
            snapshot[name] = number
        return snapshot

    def action(self, commands: Mapping[str, Any]) -> np.ndarray:
        """Convert command values keyed by channel key into the action vector.

        Raises:
            ValueError: A command is missing, unknown, or outside its bounds.
        """
        expected = {c["key"] for c in self.command_channels}
        if set(commands) != expected:
            raise ValueError(
                "commands must cover every actuator; "
                f"missing={sorted(expected - set(commands))}, "
                f"unknown={sorted(set(commands) - expected)}"
            )
        action = []
        for channel in self.command_channels:
            value = _finite(commands[channel["key"]], channel["key"])
            if not channel["low"] <= value <= channel["high"]:
                raise ValueError(
                    f"{channel['key']} must be within "
                    f"[{channel['low']}, {channel['high']}] {channel['units']}"
                )
            action.append(value)
        return np.asarray(action, dtype=float)
