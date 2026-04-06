# Observation Space

B2B environments provide a rich observation vector containing zone temperatures, outdoor weather, time features, energy consumption, and optional occupancy information. This page documents the full observation structure and how it varies across building types.

---

## Flat Observation Vector

The default observation mode produces a flat `Box` space. The observation vector is structured in the following order:

| Group | Features | Bounds | Units |
|---|---|---|---|
| Zone Air Temperatures | 1 per controlled zone | [10, 45] | °C |
| Outdoor Air Temperature | 1 | [-30, 50] | °C |
| Outdoor Air Relative Humidity | 1 | [0, 100] | % |
| Time of Day | 1 | [1, 25] | hours |
| Day of Week | 1 | [1, 7] | day (1=Sun, 7=Sat) |
| Day of Year | 1 | [1, 366] | day |
| HVAC Electricity | 1 | [0, *E*] | Wh/m² per timestep |
| HVAC Natural Gas | 1 | [0, *E*] | Wh/m² per timestep |

where *E* = 200 / `timesteps_per_hour` (≈ 16.7 at the default 5-min step, 50.0 at 15-min).

### Fixed Features (7 dimensions)

These features are always present and always in the same relative order:

| Feature | Bounds (default 5-min step) | Description |
|---|---|---|
| Outdoor Air Temperature | [-30°C, 50°C] | Site dry-bulb temperature |
| Outdoor Air Relative Humidity | [0%, 100%] | Site relative humidity |
| Time of Day | [1, 25] | Current simulation hour (EnergyPlus convention) |
| Day of Week | [1, 7] | 1=Sunday through 7=Saturday |
| Day of Year | [1, 366] | Julian day |
| HVAC Electricity | [0, 16.7] Wh/m² | Electricity consumption normalized by floor area |
| HVAC Natural Gas | [0, 16.7] Wh/m² | Gas consumption normalized by floor area |

!!! info "Energy normalization"

    Energy meters report raw Joules from EnergyPlus. B2B divides by the building's floor area and converts to Wh (÷ 3,600) to produce comparable, area-normalized values across buildings of different sizes. The energy upper bound scales with the simulation timestep: it equals 200 W/m² (assumed peak HVAC power) divided by `timesteps_per_hour`.

### Variable Features (zone temperatures)

The number of zone air temperature features varies by building:

| Building Type | Typical Zone Count | Obs Dimension |
|---|---|---|
| Single-Zone Houses | 1 | 8 |
| RestaurantFastFood | 2 | 9 |
| Warehouse | 3 | 10 |
| RetailStandalone | 4 | 11 |
| OfficeSmall | 5 | 12 |
| OfficeMedium | 15+ | 22+ |

---

## Optional Observations

### Zone Occupancy Counts

When `target_temperature_mode` is set to `"occupancy"`, the observation vector is augmented with:

| Feature | Bounds | Units |
|---|---|---|
| Zone occupancy count (per zone) | [0, 20] | people |
| Dynamic target temperature (per zone) | [10, 35] | °C |

In occupancy mode, the target temperature switches between `occupied_c` and `unoccupied_c` based on whether the zone's `Zone People Occupant Count` is greater than zero.

```python
task=TaskConfig.from_dict({
    "run_period": "winter",
    "target_temperature_mode": "occupancy",
    "default_zone_target_temperature": {
        "occupied_c": 21.0,
        "unoccupied_c": 16.0,
    },
})
```

---

## Observation Construction

Observations are assembled by the `flat_observation_info` function in `building2building.simulator.observation_spaces`. The process:

1. **Zone temperatures**: One `VariableHole("ZONE AIR TEMPERATURE", zone_name)` per controlled zone
2. **Occupancy** (optional): `DynamicZoneVariable("Zone People Occupant Count", zone_name)` per zone
3. **Target temperatures** (optional): `DynamicTargetTemperature` computed from occupancy and config
4. **Time features**: `FunctionHole` wrappers around EnergyPlus API calls for time-of-day, day-of-week, day-of-year
5. **Outdoor weather**: `VariableHole` for dry-bulb temperature and relative humidity
6. **Energy meters**: `DynamicMeter` for `Electricity:HVAC` and `NaturalGas:HVAC`, divided by floor area × 3,600

!!! note "Dynamic meters"

    The `DynamicMeter` class gracefully handles missing meters. For example, if a building has no gas-consuming equipment, the `NaturalGas:HVAC` meter may not exist — in that case, the observation value is 0.0.

---

## Dict Observation Mode

B2B also supports a structured dictionary observation space via `dict_observation_info`:

```python
{
    "temperature": {
        "ZONE 1": Box(10.0, 45.0, shape=(1,)),
        "ZONE 2": Box(10.0, 45.0, shape=(1,)),
    },
    "time": {
        "time_of_day": Box(1.0, 25.0, shape=(1,)),
        "day_of_week": Box(1.0, 7.0, shape=(1,)),
        "day_of_year": Box(1.0, 366.0, shape=(1,)),
    },
    "outdoor": {
        "temperature": Box(-30.0, 50.0, shape=(1,)),
        "humidity": Box(0.0, 100.0, shape=(1,)),
    },
    "energy": {
        "natural_gas": Box(0.0, 50.0, shape=(1,)),
        "electricity": Box(0.0, 50.0, shape=(1,)),
    },
}
```

The dict mode is primarily used internally by the reward functions, which index observations by key (e.g., `obs["energy"]["electricity"]`).

---

## How Observations Vary Across Buildings

The key challenge for multi-building generalization is that **observation dimensions vary** due to different zone counts. B2B provides two strategies:

### 1. PadObservation Wrapper

Pads zone temperatures to a fixed maximum size while keeping non-zone features at consistent indices:

```python
from building2building.simulator.wrappers import PadObservation

env = PadObservation(env, target_size=25)
# All buildings now produce 25-dimensional observations
```

Padded dimensions have bounds `[0, 0]`, so normalization layers treat them as constants. See [Wrappers](wrappers.md) for details.

### 2. AugmentObservationWithBuildingParams Wrapper

Appends building metadata (area, warmup phases, number of actuators) to help the policy identify which building it is controlling:

```python
from building2building.simulator.wrappers import AugmentObservationWithBuildingParams

env = AugmentObservationWithBuildingParams(env)
```

---

## Observation Names

Every observation feature has a human-readable name accessible via `env.metadata["observation_names"]`:

```python
meta = env.metadata
for i, name in enumerate(meta["observation_names"]):
    print(f"  [{i}] {name}")
```

Example output for a single-zone house:

```
  [0] ZONE AIR TEMPERATURE LIVING ZONE
  [1] outdoor_temperature
  [2] outdoor_humidity
  [3] time_of_day
  [4] day_of_week
  [5] day_of_year
  [6] energy_electricity
  [7] energy_gas
```

---

## Next Steps

- Understand the [action space](actions.md) structure per HVAC type
- Learn about [reward functions](rewards.md) that use these observations
- Explore [wrappers](wrappers.md) for normalizing and padding observations
