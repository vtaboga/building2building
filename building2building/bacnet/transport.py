"""BACnet/IP gateway and client for a :class:`~building2building.bacnet.points.PointStore`.

:class:`Gateway` serves every device of a store on a virtual BACnet network
behind a BACnet/IP router (BACpypes3), so standard BACnet tools can discover
and use them: Who-Is routed to the virtual network, ReadProperty and
WriteProperty. Network writes go through the store first, so the store's rules
(writable objects only, priorities 8 to 16, present-value limits) are the
gateway's. :class:`WireClient` is an asynchronous BACpypes3 client for those
devices.

Requires the ``bacnet`` extra (``pip install "building2building[bacnet]"``).
"""

# BACpypes3 builds its types with metaclasses (constructors return Sequence or
# Element as far as pyright can tell), so these checks only report noise here.
# pyright: reportArgumentType=false, reportReturnType=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import math
import re
from collections import defaultdict
from typing import Any

from bacpypes3.apdu import ErrorRejectAbortNack, SimpleAckPDU, WhoIsRequest
from bacpypes3.app import Application
from bacpypes3.basetypes import EngineeringUnits, PriorityValue
from bacpypes3.constructeddata import Array
from bacpypes3.errors import PropertyError, ServicesError
from bacpypes3.local.analog import AnalogInputObject, AnalogValueObjectCmd
from bacpypes3.local.device import DeviceObject
from bacpypes3.local.networkport import NetworkPortObject
from bacpypes3.netservice import ROUTER_AVAILABLE
from bacpypes3.pdu import Address, RemoteBroadcast
from bacpypes3.primitivedata import (
    BitString,
    Boolean,
    Enumerated,
    Null,
    ObjectIdentifier,
    Real,
)
from bacpypes3.vlan import VirtualNetwork

from building2building.bacnet.points import MAX_PRIORITY, MIN_PRIORITY, PointStore

ROUTER_INSTANCE = 4194000
CLIENT_INSTANCE = ROUTER_INSTANCE - 1
VENDOR_ID = 999
VENDOR_NAME = "Building2Building"

_UNIT_ALIASES = {
    "degc": "degreesCelsius",
    "c": "degreesCelsius",
    "°c": "degreesCelsius",
    "degrees-celsius": "degreesCelsius",
    "celsius": "degreesCelsius",
    "kg/s": "kilogramsPerSecond",
    "kg-per-second": "kilogramsPerSecond",
    "kilograms-per-second": "kilogramsPerSecond",
    "kwh": "kilowattHours",
    "kilowatt-hours": "kilowattHours",
    "people": "noUnits",
    "person": "noUnits",
    "hour": "hours",
    "day": "days",
    "unitless": "noUnits",
    "no-units": "noUnits",
    "1": "noUnits",
    "m3/s": "cubicMetersPerSecond",
    "m^3/s": "cubicMetersPerSecond",
    "cubic-meters-per-second": "cubicMetersPerSecond",
    "m/s": "metersPerSecond",
    "kw": "kilowatts",
    "w": "watts",
    "wh": "wattHours",
}

_PROPERTY_NAMES = {
    "present-value": "presentValue",
    "object-name": "objectName",
    "object-list": "objectList",
    "priority-array": "priorityArray",
    "relinquish-default": "relinquishDefault",
    "status-flags": "statusFlags",
    "object-type": "objectType",
    "object-identifier": "objectIdentifier",
    "min-pres-value": "minPresValue",
    "max-pres-value": "maxPresValue",
    "out-of-service": "outOfService",
}

_STATUS_FLAGS = ("in-alarm", "fault", "overridden", "out-of-service")

_ALL_PROPERTIES = [
    "object-identifier",
    "object-name",
    "object-type",
    "present-value",
    "description",
    "units",
    "status-flags",
    "out-of-service",
    "priority-array",
    "relinquish-default",
    "min-pres-value",
    "max-pres-value",
]


def _error_detail(exc: Exception) -> str:
    fields = [getattr(exc, name, None) for name in ("errorClass", "errorCode")]
    labels = [
        re.sub(r"(?<!^)(?=[A-Z])", "-", str(getattr(field, "attr", field))).lower()
        for field in fields
        if field is not None
    ]
    return ": ".join(labels) or str(exc)


def _units(value: str) -> EngineeringUnits:
    value = _UNIT_ALIASES.get(str(value).strip().lower(), value)
    try:
        return EngineeringUnits(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported BACnet engineering units: {value!r}") from exc


def _common(point: dict[str, Any]) -> dict[str, Any]:
    obj_type, instance = point["object"].split(",", 1)
    return {
        "objectIdentifier": (obj_type, int(instance)),
        "objectName": point["name"],
        "description": point["description"],
        "statusFlags": [0, 0, 0, 0],
        "eventState": "normal",
        "outOfService": False,
    }


def _priority_value(command: float | None) -> PriorityValue:
    return (
        PriorityValue(null=())
        if command is None
        else PriorityValue(real=float(command))
    )


def _virtual_mac(index: int) -> bytes:
    number = index + 2  # virtual MAC 1 is the router's
    if number <= 252:
        return bytes((number,))
    if number >= 1 << 24:
        raise ValueError("too many devices for a BACnet virtual network")
    return number.to_bytes(3, "big")


def _check_network_number(network_number: int) -> None:
    if (
        isinstance(network_number, bool)
        or not isinstance(network_number, int)
        or not 2 <= network_number < 65535
    ):
        raise ValueError("network_number must be an integer from 2 through 65534")


def _canonical_object_type(value: str) -> str:
    return {"analogInput": "analog-input", "analogValue": "analog-value"}.get(
        value, value
    )


def _bacnet_object_type(value: str) -> str:
    return {"analog-input": "analogInput", "analog-value": "analogValue"}.get(
        value, value
    )


def _object_identifier(object: str) -> ObjectIdentifier:
    obj_type, separator, instance = str(object).partition(",")
    if not separator:
        raise ValueError("object must be '<object-type>,<instance>'")
    return ObjectIdentifier((_bacnet_object_type(obj_type), int(instance)))


class _GatewayApplication(Application):
    """A device application whose WriteProperty goes through the point store."""

    gateway: "Gateway"

    async def do_WritePropertyRequest(self, apdu: Any) -> None:
        obj = self.get_object_id(apdu.objectIdentifier)
        point = self.gateway._point_by_object.get(id(obj))
        if point is None or not point["writable"]:
            raise PropertyError("writeAccessDenied")
        prop = getattr(apdu.propertyIdentifier, "attr", str(apdu.propertyIdentifier))
        if prop != "presentValue" or apdu.propertyArrayIndex is not None:
            raise PropertyError("writeAccessDenied")
        priority = MAX_PRIORITY if apdu.priority is None else int(apdu.priority)
        if not MIN_PRIORITY <= priority <= MAX_PRIORITY:
            raise ServicesError("parameterOutOfRange")
        try:
            value = apdu.propertyValue.cast_out(Real, null=True)
            value = None if value is None or isinstance(value, Null) else float(value)
            self.gateway._commit_write(point, value, priority)
        except ValueError as exc:
            raise PropertyError("valueOutOfRange") from exc
        except KeyError as exc:
            raise PropertyError("writeAccessDenied") from exc
        await self.response(SimpleAckPDU(context=apdu))


class _ReadOnlyApplication(Application):
    async def do_WritePropertyRequest(self, apdu: Any) -> None:
        raise PropertyError("writeAccessDenied")


class _ClientApplication(Application):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.discovered: dict[int, Address] = {}
        super().__init__(*args, **kwargs)

    async def do_IAmRequest(self, apdu: Any) -> None:
        instance = int(apdu.iAmDeviceIdentifier[1])
        if instance not in (ROUTER_INSTANCE, CLIENT_INSTANCE):
            self.discovered[instance] = apdu.pduSource
        await super().do_IAmRequest(apdu)


class Gateway:
    """Serve a point store's devices over BACnet/IP.

    A router device listens on *address* (``"host:port"``) and routes to a
    virtual BACnet network carrying one BACnet device per store device.
    Writes from the network are committed to the store as *owner*. After the
    simulator updates the store, call :meth:`sync` to mirror it into the
    BACnet objects.

    Must be created and used inside a running asyncio event loop.

    Args:
        store: The points to serve.
        address: BACnet/IP address of the router, ``"host:port"``.
        network_number: Number of the virtual network, 2 to 65534.
        owner: Label recorded with every write received from the network.
    """

    def __init__(
        self,
        store: PointStore,
        address: str,
        *,
        network_number: int = 100,
        owner: str = "client",
    ) -> None:
        _check_network_number(network_number)
        self.store, self.address, self.network_number = store, address, network_number
        self.owner = owner
        self._mirrored: dict[str, tuple[float, list[float | None]]] = {}
        self._point_by_object: dict[int, dict[str, Any]] = {}
        self._objects: dict[str, Any] = {}
        self._closed = False
        self.apps: dict[int, _GatewayApplication] = {}

        units = {point["key"]: _units(point["units"]) for point in store.points}
        self._network_name = f"b2b-{id(self):x}"
        VirtualNetwork(self._network_name)
        ip_port = NetworkPortObject(
            address,
            objectIdentifier=("network-port", 1),
            objectName="NP-IP",
            networkNumber=1,
            networkNumberQuality="configured",
        )
        virtual_port = self._virtual_port(2, b"\x01")
        self.router = _ReadOnlyApplication.from_object_list(
            [
                DeviceObject(
                    objectIdentifier=("device", ROUTER_INSTANCE),
                    objectName="B2B-ROUTER",
                    vendorIdentifier=VENDOR_ID,
                    vendorName=VENDOR_NAME,
                ),
                ip_port,
                virtual_port,
            ]
        )

        points_by_device: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for point in store.points:
            points_by_device[point["device"]].append(point)
        for index, (device, points) in enumerate(sorted(points_by_device.items())):
            objects = [
                DeviceObject(
                    objectIdentifier=("device", device),
                    objectName=store.device_name(device),
                    vendorIdentifier=VENDOR_ID,
                    vendorName=VENDOR_NAME,
                    modelName="Building2Building simulator",
                    description="Building2Building simulated BACnet device",
                ),
                self._virtual_port(1, _virtual_mac(index)),
            ]
            for point in points:
                obj = self._make_object(point, units[point["key"]])
                self._point_by_object[id(obj)] = point
                self._objects[point["key"]] = obj
                objects.append(obj)
            app = _GatewayApplication.from_object_list(objects)
            app.gateway = self
            self.apps[device] = app

    def _virtual_port(self, instance: int, mac: bytes) -> NetworkPortObject:
        return NetworkPortObject(
            objectIdentifier=("network-port", instance),
            objectName="NP-B2B",
            networkType="virtual",
            protocolLevel="bacnet-application",
            networkInterfaceName=self._network_name,
            networkNumber=self.network_number,
            networkNumberQuality="configured",
            macAddress=mac,
        )

    def _make_object(self, point: dict[str, Any], units: EngineeringUnits) -> Any:
        device, object_id = point["device"], point["object"]
        value = float(self.store.read_property(device, object_id))
        if not point["writable"]:
            return AnalogInputObject(**_common(point), presentValue=value, units=units)
        default = float(
            self.store.read_property(device, object_id, "relinquish-default")
        )
        obj = AnalogValueObjectCmd(
            **_common(point),
            presentValue=value,
            units=units,
            relinquishDefault=default,
            minPresValue=float(point["low"]),
            maxPresValue=float(point["high"]),
        )
        commands = list(self.store.read_property(device, object_id, "priority-array"))
        for index, command in enumerate(commands):
            obj.priorityArray[index] = _priority_value(command)
        self._mirrored[point["key"]] = (default, commands)
        return obj

    def _commit_write(
        self, point: dict[str, Any], value: float | None, priority: int
    ) -> None:
        self.store.write_property(
            point["device"], point["object"], value, priority=priority, owner=self.owner
        )
        self._sync_point(point)

    def _sync_point(self, point: dict[str, Any]) -> None:
        obj = self._objects[point["key"]]
        device, object_id = point["device"], point["object"]
        if not point["writable"]:
            obj.presentValue = float(self.store.read_property(device, object_id))
            return
        # Mirror only what changed: every priority-array slot set makes
        # BACpypes3 recompute presentValue.
        default = float(
            self.store.read_property(device, object_id, "relinquish-default")
        )
        commands = list(self.store.read_property(device, object_id, "priority-array"))
        old_default, old_commands = self._mirrored.get(
            point["key"], (None, [object()] * MAX_PRIORITY)
        )
        if default != old_default:
            obj.relinquishDefault = default
        for index, command in enumerate(commands):
            if command != old_commands[index]:
                obj.priorityArray[index] = _priority_value(command)
        if default != old_default:
            obj.recalculating()  # a new relinquish default alone does not trigger it
        self._mirrored[point["key"]] = (default, commands)

    def sync(self) -> None:
        """Mirror every point of the store into the BACnet objects."""
        if self._closed:
            raise RuntimeError("gateway is closed")
        for point in self.store.points:
            self._sync_point(point)

    def close(self) -> None:
        """Close the router and device applications and drop the virtual network."""
        if self._closed:
            return
        for app in self.apps.values():
            app.close()
        self.router.close()
        VirtualNetwork._networks.pop(self._network_name, None)
        self._closed = True


class WireClient:
    """Asynchronous BACnet/IP client for the devices served by a :class:`Gateway`.

    Values come back in the store's conventions: object identifiers as
    ``"analog-value,0"``, hyphenated property names, ``None`` for an empty
    priority-array slot.

    Must be created and used inside a running asyncio event loop.

    Args:
        address: BACnet/IP address of the client, ``"host:port"``.
        gateway_address: BACnet/IP address of the gateway's router.
        network_number: Number of the gateway's virtual network.
        timeout: Seconds to wait for each answer, and for I-Am answers after
            a Who-Is.
    """

    def __init__(
        self,
        address: str,
        *,
        gateway_address: str,
        network_number: int = 100,
        timeout: float = 2,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number of seconds")
        _check_network_number(network_number)
        self.address, self.gateway_address = address, gateway_address
        self.network_number, self.timeout = network_number, float(timeout)
        client_device = DeviceObject(
            objectIdentifier=("device", CLIENT_INSTANCE),
            objectName="B2B-CLIENT",
            vendorIdentifier=VENDOR_ID,
            vendorName=VENDOR_NAME,
        )
        port = NetworkPortObject(
            address,
            objectIdentifier=("network-port", 1),
            objectName="NP-CLIENT",
            networkNumber=1,
            networkNumberQuality="configured",
        )
        self.app = _ClientApplication.from_object_list([client_device, port])
        self._closed = False

    async def _read(self, address: Address, oid: ObjectIdentifier, prop: str) -> Any:
        try:
            return await asyncio.wait_for(
                self.app.read_property(address, oid, prop), self.timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"BACnet read timed out for {oid} {prop}") from exc
        except ErrorRejectAbortNack as exc:
            raise RuntimeError(
                f"BACnet read rejected for {oid} {prop}: {_error_detail(exc)}"
            ) from exc

    async def who_is(self) -> dict[str, Any]:
        """Broadcast a Who-Is on the gateway's network and list the devices that answer."""
        self._ensure_open()
        self.app.discovered.clear()
        route = Address(self.gateway_address)
        cache = self.app.nsap.router_info_cache
        await cache.set_path_info(1, self.network_number, route, ROUTER_AVAILABLE)
        await cache.set_router_dnets(1, route, {self.network_number})
        self.app.request(WhoIsRequest(destination=RemoteBroadcast(self.network_number)))
        await asyncio.sleep(self.timeout)
        devices = []
        for instance, address in sorted(self.app.discovered.items()):
            try:
                name = await self._read(
                    address, ObjectIdentifier(("device", instance)), "objectName"
                )
            except Exception:
                name = f"Device {instance}"
            devices.append(
                {
                    "device-instance": instance,
                    "address": str(address),
                    "object-name": str(name),
                }
            )
        return {"devices": devices}

    def _device_address(self, device: Any) -> tuple[int, Address]:
        try:
            instance = int(device)
        except (TypeError, ValueError):
            raise KeyError(f"unknown device: {device}") from None
        try:
            return instance, self.app.discovered[instance]
        except KeyError:
            raise KeyError(
                f"device {instance} is not discovered; call who_is() first"
            ) from None

    async def point_list(self, device: Any) -> dict[str, Any]:
        """Read a device's object list and each object's name, description, value and units."""
        self._ensure_open()
        instance, address = self._device_address(device)
        device_oid = ObjectIdentifier(("device", instance))
        rows = []
        for oid in await self._read(address, device_oid, "objectList"):
            object_type = getattr(oid[0], "attr", str(oid[0]))
            if object_type in ("device", "networkPort"):
                continue
            objid = f"{_canonical_object_type(object_type)},{oid[1]}"
            try:
                name, description, value = await asyncio.gather(
                    self._read(address, oid, "objectName"),
                    self._read(address, oid, "description"),
                    self._read(address, oid, "presentValue"),
                )
                units = (
                    await self._read(address, oid, "units")
                    if object_type.startswith("analog")
                    else "no-units"
                )
            except Exception as exc:
                raise RuntimeError(f"failed to list {objid}: {exc}") from exc
            row: dict[str, Any] = {
                "object": objid,
                "name": str(name),
                "description": str(description),
                "present-value": float(value),
                "units": str(units),
            }
            if object_type == "analogValue":
                row["commandable"] = True
            rows.append(row)
        name = await self._read(address, device_oid, "objectName")
        return {"device": instance, "object-name": str(name), "objects": rows}

    async def read_property(
        self, device: Any, object: str, property: str = "present-value"
    ) -> Any:
        """Read one property, or with ``"all"`` every property the object has."""
        self._ensure_open()
        instance, address = self._device_address(device)
        if str(property).lower() == "all":
            result = {}
            for name in _ALL_PROPERTIES:
                try:
                    result[name] = await self.read_property(instance, object, name)
                except RuntimeError as exc:
                    if "unknown-property" not in str(exc):
                        raise
            return result
        prop = _PROPERTY_NAMES.get(str(property).lower(), str(property))
        identifier = _object_identifier(object)
        try:
            value = await asyncio.wait_for(
                self.app.read_property(address, identifier, prop), self.timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"BACnet read timed out for {object} {property}"
            ) from exc
        except ErrorRejectAbortNack as exc:
            raise RuntimeError(
                f"BACnet read rejected for {object} {property}: {_error_detail(exc)}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"BACnet read failed for {object} {property}: {exc}"
            ) from exc
        if isinstance(value, Array):
            return [self._array_item(item) for item in value]
        if isinstance(value, Null):
            return None
        if isinstance(value, Boolean):
            return bool(value)
        if isinstance(value, BitString) and prop == "statusFlags":
            return {name: bool(bit) for name, bit in zip(_STATUS_FLAGS, value)}
        if isinstance(value, Enumerated):
            return str(value)
        if isinstance(value, Real):
            return float(value)
        return value if isinstance(value, (bool, int, float)) else str(value)

    @staticmethod
    def _array_item(item: Any) -> float | str | None:
        if isinstance(item, Null) or getattr(item, "null", None) is not None:
            return None
        choice = getattr(item, "_choice", None)
        if choice:
            item = getattr(item, choice)
        return float(item) if isinstance(item, Real) else str(item)

    async def write_property(
        self,
        device: Any,
        object: str,
        value: float | None,
        priority: int = MAX_PRIORITY,
    ) -> dict[str, str]:
        """Write a present value at a priority, or relinquish it with ``None``."""
        self._ensure_open()
        _instance, address = self._device_address(device)
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not MIN_PRIORITY <= priority <= MAX_PRIORITY
        ):
            raise ValueError(
                f"priority must be an integer from {MIN_PRIORITY} through {MAX_PRIORITY}"
            )
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("command value must be a finite number or None")
        identifier = _object_identifier(object)
        payload = Null(()) if value is None else value
        try:
            await asyncio.wait_for(
                self.app.write_property(
                    address, identifier, "presentValue", payload, priority=priority
                ),
                self.timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"BACnet write timed out for {object}") from exc
        except ErrorRejectAbortNack as exc:
            raise RuntimeError(
                f"BACnet write rejected for {object}: {_error_detail(exc)}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"BACnet write rejected for {object}: {exc}") from exc
        return {"result": "relinquished" if value is None else "ok"}

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("client is closed")

    def close(self) -> None:
        """Close the client application."""
        if not self._closed:
            self.app.close()
            self._closed = True
