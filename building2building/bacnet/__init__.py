"""BACnet layer for Building2Building environments.

Exposes a building's sensors and actuators as BACnet devices and objects, so
that supervisory controllers and agents can discover, read and command the
simulated building as they would a real one, in memory or over BACnet/IP.

* :class:`ChannelMap` describes an environment's observations and actuators
  as role-labelled channels.
* :class:`PointStore` turns channels into BACnet devices, analog-input
  sensors and commandable analog-value objects with priority arrays.
* :class:`BACnetBuilding` runs an environment with the point store in the
  loop and serves it over BACnet/IP.
* :class:`Gateway`, :class:`WireClient` and :class:`BACnetLink` are the
  BACnet/IP side (BACpypes3); they need the ``bacnet`` extra and are imported
  on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from building2building.bacnet.building import BACnetBuilding
from building2building.bacnet.channels import ChannelMap
from building2building.bacnet.points import PointStore

if TYPE_CHECKING:
    from building2building.bacnet.link import BACnetLink
    from building2building.bacnet.transport import Gateway, WireClient

_NETWORK = {
    "BACnetLink": "building2building.bacnet.link",
    "Gateway": "building2building.bacnet.transport",
    "WireClient": "building2building.bacnet.transport",
}

__all__ = [
    "BACnetBuilding",
    "BACnetLink",
    "ChannelMap",
    "Gateway",
    "PointStore",
    "WireClient",
]


def __getattr__(name: str) -> Any:
    if name in _NETWORK:
        import importlib

        try:
            module = importlib.import_module(_NETWORK[name])
        except ImportError as exc:
            raise ImportError(
                f"building2building.bacnet.{name} needs BACpypes3; "
                'install it with pip install "building2building[bacnet]"'
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
