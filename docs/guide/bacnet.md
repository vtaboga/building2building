# BACnet Layer

`building2building.bacnet` puts a B2B building behind BACnet, the protocol
supervisory controllers use to talk to real building automation systems.
Instead of a flat observation vector and action vector, a controller sees
BACnet devices and objects: it discovers them, reads sensors, and commands
setpoints through priority arrays, in memory or over BACnet/IP. The same
controller code can then be pointed at a simulated building or at a real
one.

```bash
pip install -e ".[bacnet]"   # adds BACpypes3, needed for BACnet/IP only
```

## Quick start

```python
import building2building as b2b
from building2building.bacnet import BACnetBuilding

env = b2b.make_env("OfficeSmall", split="test_small", index=0,
                   task="task_occ_e05", run_period="winter")
building = BACnetBuilding(env)
building.reset()

link = building.serve()                  # BACnet/IP gateway + client on loopback
devices = link.discover()["devices"]     # Who-Is
points = link.point_list(1000)           # objects of the building device

# Command one unit's supply air temperature at priority 10 for an hour
sat = next(p for p in building.store.points if p["role"] == "sat_command")
link.write(sat["device"], sat["object"], 30.0, priority=10)
for _ in range(12):
    snapshot, reward, terminated, truncated, info = building.step()
link.write(sat["device"], sat["object"], None, priority=10)   # relinquish

building.close()
```

The environment must use actions in engineering units
(`rescale_action=False`, the default).

## Point model

`PointStore` builds the devices and objects from the environment's channels
(`ChannelMap`), deterministically:

| Device | Instance | Objects |
|---|---|---|
| Building | `1000` | outdoor air temperature, HVAC electricity and gas (kWh per step), time of day, day of week, day of year |
| One per zone, in sorted zone order | `2001`, `2002`, ... | zone air temperature, occupancy, target temperature, and the zone's commands (an unconditioned zone, such as an attic, has its temperature only) |

Sensors are `analog-input` objects. Commands are commandable `analog-value`
objects with a priority array, a relinquish default and `min-pres-value` /
`max-pres-value` set to the actuator's bounds. Object names are B2B's
observation and action names, so the mapping back to the Gymnasium interface
is explicit.

| Equipment | Commands (role) | Units |
|---|---|---|
| Unitary system | supply air temperature setpoint (`sat_command`), fan air mass flow (`airflow_command`) | °C, kg/s |
| Heating-only zone equipment | heating setpoint (`heating_setpoint_command`) | °C |

This covers `OfficeSmall`, `RestaurantFastFood`, `RetailStandalone`,
`Warehouse` and `SingleFamilyHouse`. VAV systems (`OfficeMedium`) are not
mapped yet: `ChannelMap` raises `ValueError` on any observation or actuator it
does not map, rather than dropping it.

## Two layers of control

`BACnetBuilding` runs the environment the way a real building runs:

- The building's own **program** sets each command's relinquish default at
  every step. Pass it as `BACnetBuilding(env, program=fn)`, where
  `fn(snapshot)` returns `{channel key: value}`; `snapshot` maps every
  channel key to its latest value. Without a program the defaults stay at
  the middle of each actuator's range.
- **Clients** write into the priority arrays, at priorities 8 to 16 (1 to 7,
  reserved for life safety and manual operation, are refused). While a
  client holds a slot, it overrides the program; relinquishing the slot
  (writing `None`) hands control back.

At every `step()` the effective value of each command (the highest active
priority, else the relinquish default) is sent to the environment, and the
new observation is written back into the points. `store.leftover_overrides(owner)`
lists the slots a client still holds, which is useful to check that a
controller cleaned up after itself.

`reset()` keeps the same points and any networks served, restores the
relinquish defaults, and leaves the priority arrays as they are, as on a live
building.

## In memory or over BACnet/IP

`building.store` (a `PointStore`) offers the BACnet services directly, with no
network: `who_is()`, `point_list(device)`, `read_property(device, object,
property)` and `write_property(device, object, value, priority)`.

`building.serve()` starts a BACpypes3 gateway that serves the devices on a
virtual BACnet network (number 100 by default) behind a BACnet/IP router on a
free loopback UDP port, and returns a connected `BACnetLink`:

- `discover()` / `who_is()` is a Who-Is broadcast routed to the virtual network;
- `point_list(device)` reads the object list and each object's properties;
- `read(device, object, property)` and `write(device, object, value, priority)`
  are ReadProperty and WriteProperty requests.

Writes from the network go through the `PointStore`, so its rules apply on the
wire too; refusals come back as BACnet errors (`write-access-denied`,
`value-out-of-range`). Values on the wire are BACnet REALs (single
precision). The gateway listens on `link.gateway_address`, so other BACnet
tools on the same host can reach the devices as well. `BACnetLink` is
synchronous; for asyncio code, use `Gateway` and `WireClient` directly.

## Property names

Properties use BACnet's hyphenated names: `present-value`, `object-name`,
`description`, `units`, `status-flags`, `out-of-service`, `priority-array`,
`relinquish-default`, `min-pres-value`, `max-pres-value`, and `object-list` on
device objects. `read_property(..., "all")` returns every property of an
object.
