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
| Time of day | Normalized time [0, 1] |
| Day of week | Normalized day [0, 1] |
| Day of year | Normalized day [0, 1] |
| HVAC electricity | Current HVAC electric power (W/m^2) |
| HVAC gas | Current HVAC gas power (W/m^2) |

### Per-Zone Features

Each thermal zone contributes its zone air temperature (C). The number of
zone temperature observations varies by building type and instance.

### Optional Features

Depending on configuration, observations may also include:

- Zone occupancy counts (when `target_temperature_mode="occupancy"`)
- Dynamic target temperatures per zone

## Observation Names

Every observation channel has a human-readable name accessible via metadata:

```python
import building2building as b2b

env = b2b.new_make_env("OfficeSmall", task="task1")
names = env.metadata["observation_names"]
for i, name in enumerate(names):
    print(f"  [{i}] {name}")
env.close()
```

## Variable Dimensions

Observation dimensions differ across building types (different zone counts and
equipment). This heterogeneity is a core challenge of the benchmark.

| Building Type | Typical Obs Dim |
|---|---|
| `SingleFamilyHouse` | ~9 |
| `OfficeSmall` | ~12--17 |
| `OfficeMedium` | ~22--30 |
| `RetailStandalone` | ~11 |
| `RestaurantFastFood` | ~9 |
| `Warehouse` | ~10 |

## Handling Variable Dimensions

B2B provides wrappers to harmonize observations across buildings:

### `PadObservation`

Zero-pads observations to a fixed target size:

```python
env = b2b.new_make_env("OfficeSmall", task="task1")
env = b2b.PadObservation(env, target_size=40)
# obs.shape is now always (40,)
```

### `AugmentObservationWithBuildingParams`

Appends building-level parameters to the observation vector:

```python
env = b2b.new_make_env("OfficeSmall", task="task1")
env = b2b.AugmentObservationWithBuildingParams(env)
# obs now includes 5 extra features: area, n_zones, etc.
```

See [Wrappers](wrappers.md) for full details.

## Morphology Graph

For structured decomposition of the flat observation vector into per-node
local observations, see [Morphology Graph](morphology.md).
