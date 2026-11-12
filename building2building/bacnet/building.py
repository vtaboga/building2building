"""A Building2Building environment behind a BACnet point model."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Mapping

import gymnasium as gym

from building2building.bacnet.channels import ChannelMap
from building2building.bacnet.points import PointStore

if TYPE_CHECKING:
    from building2building.bacnet.link import BACnetLink

Program = Callable[[Mapping[str, float]], Mapping[str, float]]


class BACnetBuilding:
    """Run an environment whose actuators are commanded through BACnet points.

    The building has two layers of control, as a real one does:

    * its own **program**, which sets each command's relinquish default at
      every step (``program(snapshot) -> {channel key: value}``); without one
      the defaults stay where they are;
    * **clients**, which write into the commands' priority arrays (priorities
      8 to 16) through :attr:`store` or over BACnet/IP (:meth:`serve`) and
      override the program while they hold a slot.

    At every :meth:`step` the effective command values (highest active
    priority, else the relinquish default) are sent to the environment, and
    the new observation is written back into the points.

    Args:
        env: An environment from :func:`building2building.make_env`, created
            with ``rescale_action=False`` and ``sampling="zone_timestep"``.
            With the default HVAC-timestep sampling, steps vary in length,
            the ``time_of_day`` point can go backwards and the energy points
            count energy again on every extra step; a ``UserWarning`` says so.
        program: Optional building program, see above.

    Example:
        >>> import building2building as b2b
        >>> from building2building.bacnet import BACnetBuilding
        >>> env = b2b.make_env("OfficeSmall", split="test_small", index=0,
        ...                    task="task_occ_e05", run_period="winter",
        ...                    sampling="zone_timestep")
        >>> building = BACnetBuilding(env)
        >>> building.reset()
        >>> link = building.serve()          # BACnet/IP on a loopback UDP port
        >>> link.discover()["devices"]       # Who-Is
        >>> sat = next(p for p in building.store.points
        ...            if p["role"] == "sat_command")
        >>> link.write(sat["device"], sat["object"], 30.0, priority=10)
        >>> building.step()
        >>> link.write(sat["device"], sat["object"], None, priority=10)
        >>> building.close()
    """

    def __init__(self, env: gym.Env, program: Program | None = None) -> None:
        self.env = env
        self.program = program
        self.channel_map = ChannelMap(env)
        if env.metadata.get("sampling") != "zone_timestep":
            warnings.warn(
                "the environment steps at EnergyPlus HVAC system timesteps: steps "
                "vary in length, time_of_day can go backwards and energy is counted "
                "again on every extra step; create it with "
                'make_env(..., sampling="zone_timestep")',
                UserWarning,
                stacklevel=2,
            )
        self.store: PointStore | None = None
        self.snapshot: dict[str, float] = {}
        self.links: list[Any] = []
        self.done = False

    @property
    def channels(self) -> list[dict[str, Any]]:
        """The building's channels; see :class:`ChannelMap`."""
        return self.channel_map.channels

    def reset(self, **kwargs: Any) -> dict[str, float]:
        """Reset the environment and return the first snapshot.

        The first reset creates :attr:`store`. Later resets keep the same
        points, so served networks stay up; the relinquish defaults return to
        their initial values and the priority arrays are left as they are, as
        on a live building.
        """
        observation, _info = self.env.reset(**kwargs)
        snapshot = self.channel_map.snapshot(observation, reset=True)
        snapshot.update(self.channel_map.defaults())
        if self.store is None:
            self.store = PointStore(self.channels, snapshot)
        else:
            self.store.set_defaults(self.channel_map.defaults())
            self.store.update(snapshot)
            self._sync()
        self.snapshot = snapshot
        self.done = False
        return dict(snapshot)

    def step(self) -> tuple[dict[str, float], float, bool, bool, dict[str, Any]]:
        """Advance one environment step under the current commands.

        Returns:
            ``(snapshot, reward, terminated, truncated, info)``, where
            *snapshot* maps every channel key to its new value (commands to
            the value that was applied).
        """
        store = self._store()
        if self.done:
            raise RuntimeError("the episode has ended; call reset()")
        if self.program is not None:
            store.set_defaults(self.program(dict(self.snapshot)))
        commands = store.commands()
        observation, reward, terminated, truncated, info = self.env.step(
            self.channel_map.action(commands)
        )
        snapshot = self.channel_map.snapshot(observation)
        snapshot.update(commands)
        store.update(snapshot)
        self._sync()
        self.snapshot = snapshot
        self.done = bool(terminated or truncated)
        return dict(snapshot), float(reward), bool(terminated), bool(truncated), info

    def serve(self, *, owner: str = "client", **kwargs: Any) -> "BACnetLink":
        """Serve the points over BACnet/IP and return a connected client.

        Each call starts a gateway and a client of their own on free loopback
        UDP ports, on the same points; the gateway is kept in sync after every
        step and closed by :meth:`close`. Requires the ``bacnet`` extra.

        Args:
            owner: Label recorded with every write made through this link.
            **kwargs: Passed to :class:`~building2building.bacnet.link.BACnetLink`.
        """
        from building2building.bacnet.link import BACnetLink

        link = BACnetLink(self._store(), owner=owner, **kwargs)
        self.links.append(link)
        return link

    def close(self) -> None:
        """Close every served network and the environment."""
        try:
            while self.links:
                self.links.pop().close()
        finally:
            self.env.close()

    def _store(self) -> PointStore:
        if self.store is None:
            raise RuntimeError("call reset() first")
        return self.store

    def _sync(self) -> None:
        for link in self.links:
            link.sync()
