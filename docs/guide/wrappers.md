# Wrappers

B2B provides Gymnasium wrappers for observation normalization, padding, building-parameter augmentation, and multi-building training. These wrappers are essential for training RL agents that generalize across buildings.

All wrappers are located in `building2building.simulator.wrappers`.

---

## NormalizeObservation

Normalizes observations to the [0, 1] range using the observation space bounds.

### Usage

```python
from building2building.simulator.wrappers import NormalizeObservation

env = NormalizeObservation(env)
```

### Behavior

For each feature \(i\):

\[
\hat{o}_i = \frac{o_i - \text{low}_i}{\text{high}_i - \text{low}_i}
\]

- Input: raw observation from EnergyPlus (e.g., temperatures in °C, humidity in %)
- Output: observation normalized to [0, 1] based on the declared space bounds
- Values **can exceed** [0, 1] if the raw observation is outside the declared bounds

### Key Properties

| Property | Value |
|---|---|
| Observation space | `Box(0, 1, shape=original_shape)` |
| Normalization | Static, based on declared bounds |
| Denormalization | Available via `env.denormalize(obs)` |
| Bounds update | Re-reads from inner env on each `reset()` |

!!! tip "When to use"

    Always use `NormalizeObservation` when training neural network policies. Temperature features span [10, 45]°C while time features span [1, 366] — normalization ensures all features contribute equally to the policy gradient.

### Denormalization

To convert normalized observations back to physical units:

```python
raw_obs = env.denormalize(normalized_obs)
```

---

## PadObservation

Pads observations to a fixed target size with **zone-aware padding**. This is critical for training a single policy across buildings with different numbers of zones.

### Usage

```python
from building2building.simulator.wrappers import PadObservation

env = PadObservation(env, target_size=25)
```

### Behavior

The wrapper splits the observation into zone temperatures and non-zone features, pads the zone temperatures to a fixed count, then concatenates:

```
[zone_1, zone_2, ..., zone_k, 0, 0, ..., 0, outdoor_temp, humidity, time×3, energy×2]
 ├── actual zones ──┤ ├── padding ──┤ ├────── fixed features (always at same indices) ──┤
 └─────────── max_zones ────────────┘
```

| Property | Value |
|---|---|
| Target size | Configurable (e.g., 25 for up to 18 zones) |
| Max zones | `target_size - 7` (7 non-zone features) |
| Padded bounds | `[0, 0]` for padding dimensions |
| Zone detection | Uses `env.metadata["observation_names"]` |

### Why Zone-Aware Padding?

Naive zero-padding would place non-zone features (outdoor temp, time, energy) at different indices for different buildings:

```
Building A (3 zones):  [z1, z2, z3, outdoor_temp, ...]   # outdoor at index 3
Building B (5 zones):  [z1, z2, z3, z4, z5, outdoor_temp, ...]  # outdoor at index 5
```

This makes it impossible for a single policy to learn meaningful feature representations. `PadObservation` ensures non-zone features are always at the **same indices**:

```
Building A (padded):   [z1, z2, z3, 0, 0, outdoor_temp, ...]   # outdoor at index 5
Building B (padded):   [z1, z2, z3, z4, z5, outdoor_temp, ...]  # outdoor at index 5
```

!!! warning "Normalization order"

    Apply `PadObservation` **before** `NormalizeObservation`. Padded dimensions have bounds [0, 0], which would cause a division-by-zero error if normalization is applied first.

    ```python
    env = PadObservation(env, target_size=25)
    env = NormalizeObservation(env)  # Handles [0, 0] bounds correctly
    ```

---

## AugmentObservationWithBuildingParams

Appends normalized building metadata to the observation vector, enabling a policy to condition its behavior on building characteristics.

### Usage

```python
from building2building.simulator.wrappers import AugmentObservationWithBuildingParams

env = AugmentObservationWithBuildingParams(env)
```

### Augmented Parameters

| Parameter | Source | Normalization Range | Description |
|---|---|---|---|
| `area` | `env.metadata["area"]` | [50, 500] m² | Building floor area |
| `warmup_phases` | `env.metadata["warmup_phases"]` | [1, 10] | Simulation warmup phases |
| `num_actuators` | `len(env.metadata["hvac_actuators"])` | [1, 20] | Number of HVAC actuators |
| `year_built` | `building_source_metadata["year_built"]` | [1940, 2025] | Construction year |
| `num_units` | `building_source_metadata["geometry_building_num_units"]` | [1, 10] | Number of building units |

All parameters are normalized to [-1, 1]:

\[
\hat{p} = 2 \cdot \frac{p - p_{\min}}{p_{\max} - p_{\min}} - 1
\]

Values outside the expected range are clipped with a warning.

### Key Properties

| Property | Value |
|---|---|
| Added dimensions | 5 |
| Normalization range | [-1, 1] |
| Observation space | Original + 5 dimensions |
| Default values | Used when metadata is unavailable |
| Re-extraction | On each `reset()` (for multi-building training) |

### Denormalization

```python
raw_obs = env.denormalize(augmented_obs)
# Returns observation without building params, optionally denormalized by inner wrapper
```

---

## ResampleBuildingOnResetWrapper

Resamples a new building environment on each episode reset, enabling **multi-task training** across many buildings with a single environment instance.

### Usage

```python
from building2building.simulator.wrappers import ResampleBuildingOnResetWrapper

def env_factory(index: int) -> gym.Env:
    return make_single_zone_env(
        split="train",
        split_index=index,
        eplus_output_dir=f"outputs/eplus/{index}",
    )

env = ResampleBuildingOnResetWrapper(
    env_factory=env_factory,
    available_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    wandb_prefix="train",
    log_interval_steps=2048,
)
```

### Behavior

On each call to `reset()`:

1. Log the previous episode's summary (reward, length) to W&B
2. Sample a new building index uniformly from `available_indices`
3. If the index differs from the current one, close the old environment and create a new one via `env_factory`
4. Reset the new environment and log the new building's parameters to W&B
5. Reset episode counters

### Key Properties

| Property | Value |
|---|---|
| Sampling | Uniform random from `available_indices` |
| Environment lifecycle | Old env closed, new env created on index change |
| Episode tracking | Reward, steps, and episode count tracked |
| W&B logging | Episode summaries + building params + intermediate rewards |
| Error handling | `IndexError` during `step()` triggers automatic resampling |

### W&B Logging

The wrapper logs the following to W&B (best-effort, never raises):

| Metric | When | Description |
|---|---|---|
| `{prefix}/episode/reward` | On reset | Total reward of completed episode |
| `{prefix}/episode/length` | On reset | Steps in completed episode |
| `{prefix}/episode/number` | On reset | Cumulative episode count |
| `{prefix}/episode/building_index` | On reset | Split index of the building |
| `{prefix}/building/split_index` | After reset | New building's split index |
| `{prefix}/building/id` | After reset | New building's database ID |
| `{prefix}/episode/cumulative_reward` | Every N steps | Mid-episode reward for long episodes |

### Combining Wrappers

A typical multi-building training setup:

```python
from building2building.api import make_single_zone_env
from building2building.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
    PadObservation,
    ResampleBuildingOnResetWrapper,
)

def env_factory(index: int):
    env = make_single_zone_env(
        split="train",
        split_index=index,
        eplus_output_dir=f"outputs/eplus/{index}",
    )
    env = PadObservation(env, target_size=25)
    env = AugmentObservationWithBuildingParams(env)
    env = NormalizeObservation(env)
    return env

env = ResampleBuildingOnResetWrapper(
    env_factory=env_factory,
    available_indices=list(range(100)),
    wandb_prefix="train",
)
```

!!! warning "Wrapper ordering"

    The recommended wrapper ordering from inner to outer:

    1. `PadObservation` — Pad zones to fixed size
    2. `AugmentObservationWithBuildingParams` — Add building metadata
    3. `NormalizeObservation` — Normalize to [0, 1]
    4. `ResampleBuildingOnResetWrapper` — Multi-building resampling (outermost)

    `ResampleBuildingOnResetWrapper` should be the outermost wrapper since it manages the full environment lifecycle. The inner wrappers should be applied inside `env_factory`.

---

## Next Steps

- Configure wrappers via [Hydra configuration](configuration.md)
- Learn about [reward functions](rewards.md) that drive the optimization
- Set up multi-building training in [Getting Started](../getting-started.md)
