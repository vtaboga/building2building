# Wrappers

B2B provides four Gymnasium wrappers for multi-building training and
observation processing. All are exported from the top-level `building2building`
package.

## NormalizeObservation

Running-mean normalization of observations. Tracks a moving mean and variance
and normalizes each observation channel to approximately zero mean and unit
variance.

```python
import building2building as b2b

env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.NormalizeObservation(env)
```

The wrapper updates statistics on each `step()` call. To freeze statistics
at evaluation time, set `env.training = False`.

**Denormalization** is available via `env.denormalize(obs)`.

## PadObservation

Zero-pads observations to a uniform size. Required for training across buildings
with different observation dimensions.

```python
env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.PadObservation(env, target_size=40)
# obs.shape is always (40,) regardless of the building
```

Padding is zone-aware: it pads the zone-specific portion of the observation
vector, preserving the global features at their original positions.

## AugmentObservationWithBuildingParams

Appends building-level metadata to the observation vector. This enables
policies to condition on building properties for better generalization.

```python
env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.AugmentObservationWithBuildingParams(env)
```

**Appended features** (5 dimensions, normalized):

| Feature | Description |
|---|---|
| Conditioned area | Net conditioned floor area |
| Number of zones | Total thermal zones |
| Number of controlled zones | Zones with HVAC actuators |
| Observation dim | Original observation dimension |
| Action dim | Action space dimension |

## ResampleBuildingOnResetWrapper

Samples a new building from a pool on each `reset()` call. Essential for
multi-building training where the agent should generalize across instances.

```python
import building2building as b2b

def factory(idx: int) -> gym.Env:
    return b2b.new_make_env("OfficeSmall", split="train", index=idx, task="task_const_e0")

env = b2b.ResampleBuildingOnResetWrapper(
    factory,
    available_indices=[0, 1, 2, 3, 4],
)
# Each env.reset() creates a new building environment from the pool
```

## Recommended Wrapper Order

When combining wrappers, apply them in this order:

```python
env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.PadObservation(env, target_size=40)
env = b2b.AugmentObservationWithBuildingParams(env)
env = b2b.NormalizeObservation(env)
```

1. **PadObservation** first (fixes dimension)
2. **AugmentObservationWithBuildingParams** (adds building features)
3. **NormalizeObservation** last (normalizes the complete vector)

For multi-building training, wrap the factory function with
`ResampleBuildingOnResetWrapper` and apply the observation wrappers inside
the factory.
