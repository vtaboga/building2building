# Wrappers

B2B exposes wrapper primitives for observation preprocessing, cross-building
training, and RL-oriented action/observation interfaces.

## NormalizeObservation

`NormalizeObservation` performs deterministic affine scaling from the wrapped
environment bounds to `[0, 1]`:

`normalized = (obs - low) / (high - low)`.

```python
import building2building as b2b

env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.NormalizeObservation(env)
```

`denormalize()` maps normalized observations back to the wrapped space.

### Tested invariants

- Mid-range affine math is exact (for example, midpoint maps to `0.5`).
- `denormalize(observation(x))` round-trips to `x` within float tolerance.
- Zero-range features (`low == high`) raise `ValueError`.
- `reset()` rebuilds bounds if an inner wrapper swapped the environment.
- Test file: [tests/quick/test_normalize_observation.py](https://github.com/placeholder/Building2Building/blob/main/tests/quick/test_normalize_observation.py)

## PadObservation

`PadObservation` pads variable-size observations to a fixed target size for
multi-building training. Padding is zone-aware: zone temperatures are padded
first and non-zone features stay grouped at the end.

```python
env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.PadObservation(env, target_size=40)
```

### Tested invariants

- Zone slots are inferred from `metadata["observation_names"]` when present.
- Zone-name matching is case-insensitive and whitespace-tolerant.
- Non-zone features are preserved at the tail of the padded vector.
- `reset()` rebuilds split logic if shape/metadata changes after resampling.
- Test file: [tests/quick/test_pad_observation.py](https://github.com/placeholder/Building2Building/blob/main/tests/quick/test_pad_observation.py)

## AugmentObservationWithBuildingParams

`AugmentObservationWithBuildingParams` appends normalized building metadata
features to each observation.

```python
env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.AugmentObservationWithBuildingParams(env)
```

By default (`allow_defaults=False`), missing required metadata raises
`KeyError`. Set `allow_defaults=True` to opt in to legacy default filling.

### Tested invariants

- Complete metadata produces finite normalized params in `[-1, 1]`.
- Missing metadata raises `KeyError` by default.
- `allow_defaults=True` preserves warning + default behavior.
- `reset()` re-extracts metadata and rebuilds the observation space.
- `denormalize()` strips appended params and delegates to the inner wrapper.
- Test file: [tests/quick/test_augment_building_params.py](https://github.com/placeholder/Building2Building/blob/main/tests/quick/test_augment_building_params.py)

## ResampleBuildingOnResetWrapper

`ResampleBuildingOnResetWrapper` samples from `available_indices` on `reset()`
and recreates the inner environment when the sampled index changes.

```python
import gymnasium as gym
import building2building as b2b

def factory(idx: int) -> gym.Env:
    return b2b.new_make_env("OfficeSmall", split="train", index=idx, task="task_const_e0")

env = b2b.ResampleBuildingOnResetWrapper(factory, available_indices=[0, 1, 2, 3, 4])
```

The `step()` `IndexError` path is intentional: it emits `RuntimeWarning`,
returns a terminal transition with zero reward, and defers resampling to the
next `reset()`.

### Tested invariants

- Empty `available_indices` raises `ValueError`.
- Single-index pools do not recreate environments across resets.
- Multi-index pools recreate envs and close the previous one on swap.
- Episode counters reset correctly across episodes.
- `IndexError` path warns and terminates, then resamples on next `reset()`.
- W&B logging is a no-op when inactive.
- Test file: [tests/quick/test_resample_building_wrapper.py](https://github.com/placeholder/Building2Building/blob/main/tests/quick/test_resample_building_wrapper.py)

## wrap_env_for_rl

`wrap_env_for_rl` is the canonical RL wrapper composition helper:

- optional `RescaleAction(..., -1, 1)` (inner)
- optional `NormalizeObservation(...)` (outer)

```python
import building2building as b2b

env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)
```

### Tested invariants

- Composition order is stable (`RescaleAction` inner, `NormalizeObservation` outer).
- Action round-trip maps policy actions in `[-1, 1]` to engineering-unit bounds.
- `normalize_obs` and `rescale_action` flags are independent.
- Wrapped env metadata remains accessible via wrapper fallthrough.
- Test file: [tests/quick/test_wrap_env_for_rl.py](https://github.com/placeholder/Building2Building/blob/main/tests/quick/test_wrap_env_for_rl.py)

## Recommended wrapper order

For explicit composition outside `wrap_env_for_rl`, use:

```python
env = b2b.new_make_env("OfficeSmall", task="task_const_e0")
env = b2b.PadObservation(env, target_size=40)
env = b2b.AugmentObservationWithBuildingParams(env)
env = b2b.NormalizeObservation(env)
```

For RL training/evaluation, prefer `wrap_env_for_rl(...)` for action/observation
wrapping and add `PadObservation` / `AugmentObservationWithBuildingParams` as
needed for multi-building setups.
