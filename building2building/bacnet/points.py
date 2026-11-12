"""In-memory BACnet point model with deterministic object IDs and priority arrays.

:class:`PointStore` turns a list of role-labelled channels (see
:mod:`building2building.bacnet.channels`) into BACnet devices and objects:

* one building device (instance ``1000``) holding the channels that belong to
  no zone (outdoor temperature, HVAC meters, the clock);
* one device per zone (instances ``2001``, ``2002``, ... in sorted zone order)
  holding that zone's sensors and commands.

Sensors are ``analog-input`` objects. Commands are commandable
``analog-value`` objects with a 16-slot priority array, a relinquish default
and present-value limits. The present value of a command is the value at the
highest active priority, or the relinquish default when every slot is empty.
The relinquish default is where the building's own program writes (see
:meth:`PointStore.set_defaults`); clients write into the priority array.

The store is synchronous and network-free. :mod:`building2building.bacnet.transport`
serves the same store over BACnet/IP.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping

BUILDING_DEVICE = 1000
FIRST_ZONE_DEVICE = 2001
COMMAND_ROLES = frozenset(
    {"sat_command", "airflow_command", "heating_setpoint_command"}
)
MIN_PRIORITY = 8
MAX_PRIORITY = 16


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _device_number(device: Any) -> int:
    try:
        return int(device)
    except (TypeError, ValueError):
        raise KeyError(f"unknown device: {device}") from None


class PointStore:
    """BACnet devices and objects backed by a dictionary of channel values.

    Args:
        channels: Channel descriptions, each a mapping with at least ``key``.
            Optional fields: ``name``, ``description``, ``units``, ``role``,
            ``zone`` and, for commands, ``writable``, ``low`` and ``high``.
            Only channels whose ``role`` is in :data:`COMMAND_ROLES` may be
            writable.
        snapshot: Initial value of every channel, keyed by channel key. The
            initial value of a command becomes its relinquish default.

    Attributes:
        points: One row per BACnet object, in deterministic order. Each row
            has ``device``, ``object`` (``"<object-type>,<instance>"``),
            ``key``, ``name``, ``description``, ``units``, ``role``, ``zone``,
            ``writable``, ``low`` and ``high``.
        events: Log of every write and relinquish, in order.
    """

    def __init__(
        self, channels: Iterable[Mapping[str, Any]], snapshot: Mapping[str, Any]
    ) -> None:
        channels = list(channels)
        self._channels: dict[str, dict[str, Any]] = {}
        self._values: dict[str, float] = {}
        self._defaults: dict[str, float] = {}
        self._arrays: dict[str, list[float | None]] = {}
        self._owners: dict[str, dict[int, str]] = {}
        self.events: list[dict[str, Any]] = []
        self.points: list[dict[str, Any]] = []

        zones = sorted({str(c["zone"]) for c in channels if c.get("zone") is not None})
        self._zone_devices = {
            zone: FIRST_ZONE_DEVICE + i for i, zone in enumerate(zones)
        }
        self._device_names = {BUILDING_DEVICE: "B2B Building"}
        self._device_names.update({v: z for z, v in self._zone_devices.items()})

        counts: dict[tuple[int, str], int] = defaultdict(int)
        ordered = sorted(
            channels,
            key=lambda c: (
                str(c.get("zone") or ""),
                str(c.get("role", "")),
                str(c.get("name", "")),
                str(c["key"]),
            ),
        )
        for channel in ordered:
            self._add(channel, snapshot, counts)

    def _add(
        self,
        channel: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        counts: dict[tuple[int, str], int],
    ) -> None:
        key = channel.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("channel key must be a non-empty string")
        if key in self._channels:
            raise ValueError(f"duplicate channel key: {key}")
        if key not in snapshot:
            raise KeyError(f"snapshot is missing channel {key}")
        value = _number(snapshot[key], f"snapshot value for {key}")
        role = str(channel.get("role", ""))
        writable = bool(channel.get("writable", False))
        if writable and role not in COMMAND_ROLES:
            raise ValueError(f"unsupported writable channel role: {role or '<empty>'}")
        zone = channel.get("zone")
        device = BUILDING_DEVICE if zone is None else self._zone_devices[str(zone)]
        obj_type = "analog-value" if writable else "analog-input"
        instance = counts[(device, obj_type)]
        counts[(device, obj_type)] += 1
        name = str(channel.get("name") or key)
        row: dict[str, Any] = {
            "device": device,
            "object": f"{obj_type},{instance}",
            "key": key,
            "name": name,
            "description": str(channel.get("description") or name),
            "units": str(channel.get("units") or "no-units"),
            "role": role,
            "zone": zone,
            "writable": writable,
            "low": channel.get("low"),
            "high": channel.get("high"),
        }
        if writable:
            low = _number(channel.get("low"), f"low bound for {key}")
            high = _number(channel.get("high"), f"high bound for {key}")
            if low > high or not low <= value <= high:
                raise ValueError(f"invalid bounds or initial value for {key}")
            row["low"], row["high"] = low, high
            self._defaults[key] = value
            self._arrays[key] = [None] * MAX_PRIORITY
            self._owners[key] = {}
        self._channels[key] = row
        self._values[key] = value
        self.points.append(row)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _point(self, device: Any, object: str) -> dict[str, Any]:
        device = _device_number(device)
        for point in self.points:
            if point["device"] == device and point["object"] == str(object):
                return point
        raise KeyError(f"unknown object {object!r} on device {device}")

    def _check_device(self, device: Any) -> int:
        device = _device_number(device)
        if device not in self._device_names:
            raise KeyError(f"unknown device: {device}")
        return device

    def _present_value(self, point: Mapping[str, Any]) -> float:
        key = point["key"]
        if point["writable"]:
            return next(
                (v for v in self._arrays[key] if v is not None), self._defaults[key]
            )
        return self._values[key]

    def _active_priority(self, point: Mapping[str, Any]) -> int | None:
        if not point["writable"]:
            return None
        return next(
            (i for i, v in enumerate(self._arrays[point["key"]], 1) if v is not None),
            None,
        )

    def _status_flags(self, point: Mapping[str, Any]) -> dict[str, bool]:
        return {
            "in-alarm": False,
            "fault": False,
            "overridden": self._active_priority(point) is not None,
            "out-of-service": False,
        }

    def _device_object(self, device: int) -> dict[str, Any]:
        objects = [p["object"] for p in self.points if p["device"] == device]
        return {
            "object-identifier": f"device,{device}",
            "object-name": self._device_names[device],
            "object-type": "device",
            "description": "Building2Building simulated BACnet device",
            "vendor-name": "Building2Building",
            "model-name": "Building2Building simulator",
            "location": "Building2Building",
            "system-status": "operational",
            "protocol-revision": 22,
            "object-list": [f"device,{device}"] + objects,
        }

    # ------------------------------------------------------------------
    # BACnet services
    # ------------------------------------------------------------------

    def device_name(self, device: Any) -> str:
        """Return the ``object-name`` of a device."""
        return self._device_names[self._check_device(device)]

    def who_is(self) -> dict[str, Any]:
        """Return every device, as an I-Am answer to a global Who-Is would."""
        devices = [
            {
                "device-instance": d,
                "object-name": name,
                "vendor-name": "Building2Building",
            }
            for d, name in sorted(self._device_names.items())
        ]
        return {"devices": devices}

    def point_list(self, device: Any) -> dict[str, Any]:
        """Return the objects of a device with their name, description, value and units."""
        device = self._check_device(device)
        rows = []
        for p in self.points:
            if p["device"] != device:
                continue
            row: dict[str, Any] = {
                "object": p["object"],
                "name": p["name"],
                "description": p["description"],
                "present-value": self._present_value(p),
                "units": p["units"],
            }
            if p["writable"]:
                row["commandable"] = True
            rows.append(row)
        return {
            "device": device,
            "object-name": self._device_names[device],
            "objects": rows,
        }

    def read_property(
        self, device: Any, object: str, property: str = "present-value"
    ) -> Any:
        """Read one property of an object, or every property with ``"all"``.

        Property names are BACnet's hyphenated names (``present-value``,
        ``priority-array``, ``relinquish-default``, ...); underscores and
        spaces are accepted in their place.

        Raises:
            KeyError: Unknown device, object or property.
        """
        prop = str(property).strip().lower().replace("_", "-").replace(" ", "-")
        device = self._check_device(device)
        if str(object) == f"device,{device}":
            props = self._device_object(device)
            if prop == "all":
                return props
            if prop not in props:
                raise KeyError(f"unknown property: {property}")
            return deepcopy(props[prop])
        if prop == "object-list":
            raise KeyError("object-list is available only on the device object")

        p = self._point(device, object)
        common = {
            "object-identifier": p["object"],
            "object-name": p["name"],
            "object-type": p["object"].split(",", 1)[0],
            "present-value": self._present_value(p),
            "description": p["description"],
            "units": p["units"],
            "status-flags": self._status_flags(p),
            "out-of-service": False,
        }
        commands = {}
        if p["writable"]:
            commands = {
                "priority-array": list(self._arrays[p["key"]]),
                "relinquish-default": self._defaults[p["key"]],
                "min-pres-value": p["low"],
                "max-pres-value": p["high"],
            }
        if prop == "all":
            return {**common, **commands}
        if prop in common:
            return common[prop]
        if prop in commands:
            return commands[prop]
        if prop in (
            "priority-array",
            "relinquish-default",
            "min-pres-value",
            "max-pres-value",
        ):
            raise KeyError(f"{prop} is available only for commands")
        raise KeyError(f"unknown property: {property}")

    def write_property(
        self,
        device: Any,
        object: str,
        value: float | None,
        priority: int = MAX_PRIORITY,
        owner: str = "client",
    ) -> dict[str, Any]:
        """Write a command's present value at a priority, or relinquish it with ``None``.

        Priorities 1 to 7 (life safety and manual operator levels) are refused.

        Args:
            device: Device instance.
            object: Object identifier, e.g. ``"analog-value,0"``.
            value: Value in the command's units, within its present-value
                limits, or ``None`` to relinquish the slot.
            priority: Priority-array slot, 8 to 16.
            owner: Label recorded with the write, used by
                :meth:`leftover_overrides`.

        Raises:
            KeyError: Unknown device or object.
            ValueError: Read-only object, priority out of range, or value
                outside the command's limits.
        """
        p = self._point(device, object)
        if not p["writable"]:
            raise ValueError(f"{object} is read-only")
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not MIN_PRIORITY <= priority <= MAX_PRIORITY
        ):
            raise ValueError(
                f"priority must be an integer from {MIN_PRIORITY} through {MAX_PRIORITY}"
            )
        key = p["key"]
        if value is None:
            self._arrays[key][priority - 1] = None
            self._owners[key].pop(priority, None)
            result = "relinquished"
        else:
            number = _number(value, "command value")
            if not p["low"] <= number <= p["high"]:
                raise ValueError(
                    f"value {number} is outside [{p['low']}, {p['high']}] for {p['name']}"
                )
            self._arrays[key][priority - 1] = number
            self._owners[key][priority] = str(owner)
            value, result = number, "ok"
        self.events.append(
            {
                "device": p["device"],
                "object": p["object"],
                "owner": str(owner),
                "priority": priority,
                "value": value,
                "active-value": self._present_value(p),
                "active-priority": self._active_priority(p),
                "result": result,
            }
        )
        return {"result": result, "present-value": self._present_value(p)}

    # ------------------------------------------------------------------
    # Simulation side
    # ------------------------------------------------------------------

    def update(self, snapshot: Mapping[str, Any]) -> None:
        """Replace every channel's value with a new snapshot from the simulator."""
        for key in self._channels:
            if key not in snapshot:
                raise KeyError(f"snapshot is missing channel {key}")
            self._values[key] = _number(snapshot[key], f"snapshot value for {key}")

    def set_defaults(self, commands: Mapping[str, Any]) -> None:
        """Set relinquish defaults: what the building's own program commands.

        Args:
            commands: Command values keyed by channel key; channels not listed
                keep their current default.
        """
        for key, value in commands.items():
            if key not in self._defaults:
                raise KeyError(f"unknown command channel: {key}")
            number = _number(value, f"command default for {key}")
            p = self._channels[key]
            if not p["low"] <= number <= p["high"]:
                raise ValueError(
                    f"default for {key} is outside [{p['low']}, {p['high']}]"
                )
            self._defaults[key] = number

    def commands(self) -> dict[str, float]:
        """Return the effective value of every command, keyed by channel key."""
        return {key: self._present_value(self._channels[key]) for key in self._defaults}

    def leftover_overrides(self, owner: str = "client") -> list[dict[str, Any]]:
        """List the priority-array slots still held by *owner*."""
        result = []
        for p in self.points:
            if not p["writable"]:
                continue
            key = p["key"]
            for priority, value in enumerate(self._arrays[key], 1):
                if value is not None and self._owners[key].get(priority) == owner:
                    result.append(
                        {
                            "device": p["device"],
                            "object": p["object"],
                            "name": p["name"],
                            "priority": priority,
                            "value": value,
                        }
                    )
        return result
