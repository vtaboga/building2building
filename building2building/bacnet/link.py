"""Synchronous BACnet/IP access to a point store, for code that is not async.

:class:`BACnetLink` starts a :class:`~building2building.bacnet.transport.Gateway`
and a :class:`~building2building.bacnet.transport.WireClient` on free UDP
ports of one host and runs both on an asyncio event loop in a background
thread. Every :meth:`~BACnetLink.read` and :meth:`~BACnetLink.write` is a
ReadProperty or WriteProperty request on the wire, and discovery is a Who-Is
routed to the gateway's virtual network; the calls block until the answer
comes back. The simulation stays in the caller's thread: call
:meth:`~BACnetLink.sync` after each step to mirror the store into the gateway
(:class:`~building2building.bacnet.building.BACnetBuilding` does this).

Requires the ``bacnet`` extra (``pip install "building2building[bacnet]"``).
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Coroutine, TypeVar

from building2building.bacnet.points import MAX_PRIORITY, PointStore
from building2building.bacnet.transport import Gateway, WireClient

CALL_TIMEOUT_S = 60.0

_T = TypeVar("_T")


def _free_ports(n: int, host: str) -> list[int]:
    socks = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((host, 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


class BACnetLink:
    """A gateway serving *store* and a client connected to it.

    Args:
        store: The points to serve.
        owner: Label recorded with every write made through this link.
        host: Host to bind both UDP ports on.
        timeout: Seconds the client waits for each answer and for I-Am
            answers after a Who-Is.
        network_number: Number of the gateway's virtual network.

    Attributes:
        gateway_address: ``"host:port"`` of the gateway's BACnet/IP router;
            other BACnet clients on the host can use it too.
        requests: Number of requests made through this link.
    """

    def __init__(
        self,
        store: PointStore,
        *,
        owner: str = "client",
        host: str = "127.0.0.1",
        timeout: float = 3.0,
        network_number: int = 100,
    ) -> None:
        self.store, self.requests = store, 0
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="bacnet-link", daemon=True
        )
        self._thread.start()
        self.gateway: Gateway | None = None
        self.client: WireClient | None = None
        try:
            gateway_port, client_port = _free_ports(2, host)
            self.gateway_address = f"{host}:{gateway_port}"
            self.client_address = f"{host}:{client_port}"

            async def start() -> tuple[Gateway, WireClient]:
                gateway = Gateway(
                    store,
                    self.gateway_address,
                    network_number=network_number,
                    owner=owner,
                )
                client = WireClient(
                    self.client_address,
                    gateway_address=self.gateway_address,
                    network_number=network_number,
                    timeout=timeout,
                )
                return gateway, client

            self.gateway, self.client = self._call(start())
        except Exception:
            self.close()
            raise

    def _call(self, coro: Coroutine[Any, Any, _T]) -> _T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(CALL_TIMEOUT_S)

    def _client(self) -> WireClient:
        if self.client is None:
            raise RuntimeError("link is closed")
        return self.client

    def who_is(self) -> dict[str, Any]:
        """Broadcast a Who-Is and list the devices that answer."""
        self.requests += 1
        return self._call(self._client().who_is())

    def discover(self, attempts: int = 3) -> dict[str, Any]:
        """Who-Is until every device of the store has answered.

        A freshly started stack may miss the first broadcast while it binds
        its devices, so the Who-Is is repeated up to *attempts* times.
        """
        want = {p["device"] for p in self.store.points}
        found: dict[str, Any] = {"devices": []}
        for _ in range(attempts):
            found = self.who_is()
            if want <= {d["device-instance"] for d in found["devices"]}:
                return found
        seen = sorted(d["device-instance"] for d in found["devices"])
        raise RuntimeError(f"BACnet discovery found {seen}, expected {sorted(want)}")

    def point_list(self, device: Any) -> dict[str, Any]:
        """List a device's objects with their name, description, value and units."""
        self.requests += 1
        return self._call(self._client().point_list(device))

    def read(self, device: Any, object: str, prop: str = "present-value") -> Any:
        """ReadProperty; *prop* ``"all"`` reads every property the object has."""
        self.requests += 1
        return self._call(self._client().read_property(device, object, prop))

    def write(
        self,
        device: Any,
        object: str,
        value: float | None,
        priority: int = MAX_PRIORITY,
    ) -> dict[str, str]:
        """WriteProperty on present-value at *priority*; ``None`` relinquishes."""
        self.requests += 1
        return self._call(
            self._client().write_property(device, object, value, priority=priority)
        )

    def sync(self) -> None:
        """Mirror the store into the gateway's BACnet objects."""
        gateway = self.gateway
        if gateway is None:
            raise RuntimeError("link is closed")

        async def run() -> None:
            gateway.sync()

        self._call(run())

    def close(self) -> None:
        """Close the client and the gateway and stop the event loop."""

        async def stop() -> None:
            for part in (self.client, self.gateway):
                if part is not None:
                    part.close()
            await asyncio.sleep(0)

        try:
            if self._loop.is_running():
                self._call(stop())
        finally:
            self.client = self.gateway = None
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if not self._loop.is_running():
                self._loop.close()
