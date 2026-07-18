# Observation Space

## Structure

Observations are flat `numpy` arrays combining global features shared across
all buildings with zone-specific features that vary per building.

### Global Features

Present in every environment:

| Feature | Description |
|---|---|
| Outdoor air temperature | Current outdoor dry-bulb temperature (C) |
| Outdoor humidity | Current outdoor relative humidity (%) |
| Time of day | Fractional hour of day, in (0, 24] |
| Day of week | 1--7 (EnergyPlus convention: 1 = Sunday) |
| Day of year | 1--366 |
| HVAC electricity | HVAC electricity use over the last timestep, per floor area (Wh/m^2/timestep) |
| HVAC gas | HVAC natural gas use over the last timestep, per floor area (Wh/m^2/timestep) |

### Per-Zone Features

Each thermal zone contributes its zone air temperature (C). The number of
zone temperature observations varies by building type and instance.

### Optional Features

Depending on configuration, observations may also include (per controlled
zone):

- Zone occupancy counts (when `target_temperature_mode` is `"occupancy"` or
  `"random_schedule"`)
- Dynamic target temperatures (same modes)

## Observation Names

Every observation channel has a human-readable name accessible via metadata:

```python
import building2building as b2b

env = b2b.make_env("OfficeSmall", task="task_const_e0")
names = env.metadata["observation_names"]
for i, name in enumerate(names):
    print(f"  [{i}] {name}")
env.close()
```

## Variable Dimensions

Observation dimensions differ across building types (different zone counts and
equipment). This heterogeneity is a core challenge of the benchmark.

For constant-setpoint tasks, the observation is the zone air temperatures plus
the 7 global features above:

| Building Type | Zones | Obs Dim (constant tasks) |
|---|---|---|
| `SingleFamilyHouse` | 2--4 | 9--11 |
| `OfficeSmall` | 6 | 13 |
| `OfficeMedium` | 18 | 25 |
| `RetailStandalone` | 5 | 12 |
| `RestaurantFastFood` | 3 | 10 |
| `Warehouse` | 3 | 10 |

Occupancy-based and random-schedule tasks add two more features per
*controlled* zone (occupancy count and dynamic target temperature).

## Handling Variable Dimensions

B2B provides wrappers to harmonize observations across buildings:

### `PadObservation`

Zero-pads observations to a fixed target size:

```python
env = b2b.make_env("OfficeSmall", task="task_const_e0")
env = b2b.PadObservation(env, target_size=40)
# obs.shape is now always (40,)
```

### `AugmentObservationWithBuildingParams`

Appends building-level parameters to the observation vector:

```python
env = b2b.make_env("OfficeSmall", task="task_const_e0")
env = b2b.AugmentObservationWithBuildingParams(env)
# obs now includes 5 extra features: area, warmup_phases,
# num_actuators, year_built, num_units
```

See [Wrappers](wrappers.md) for full details.

## Morphology Graph

For structured decomposition of the flat observation vector into per-node
local observations, see [Morphology Graph](morphology.md).
