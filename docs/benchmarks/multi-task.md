# Multi-Task Learning Benchmark

## Task Definition

Train a **single policy** on **N** buildings simultaneously, where each
episode is drawn from the training pool.  The objective is to learn a policy
that performs well across all buildings without per-building fine-tuning.

## Core Mechanism: `ResampleBuildingOnResetWrapper`

```python
from building2building.simulator.wrappers import ResampleBuildingOnResetWrapper
```

`ResampleBuildingOnResetWrapper` wraps a Gymnasium environment and, on every
call to `reset()`, uniformly samples a new building index from the available
pool.  If the new index differs from the current one, the old environment is
closed and a fresh one is created via the provided `env_factory`.

This makes the multi-task setup transparent to the training loop — the agent
simply calls `env.reset()` and receives observations from a potentially
different building each episode.

## Observation Harmonisation Wrappers

Because different buildings have different numbers of zones (and therefore
different observation sizes), two wrappers normalise the observation space:

### `PadObservation`

Pads observations to a fixed target size with zone-aware padding.  Zone
temperatures are padded to a configurable maximum while shared features
(outdoor temperature, time-of-day, energy) are kept in consistent positions.

```python
from building2building.simulator.wrappers import PadObservation
```

### `AugmentObservationWithBuildingParams`

Appends normalised building parameters (floor area, number of zones, etc.) to
the observation vector so the policy can condition its behaviour on building
characteristics.

```python
from building2building.simulator.wrappers import AugmentObservationWithBuildingParams
```

## Configuration

The multi-task training script is `scripts/train_multizones.py`, driven by
`configs/train_multizones.yaml`.

```yaml title="configs/train_multizones.yaml (excerpt)"
defaults:
  - _self_
  - bldg: multi_zone           # building_type, split, index live here
  - ...

env:
  normalize_obs: true
  normalize_action: true
  max_steps: 35040              # full year at 15-min timesteps

seed: 42
```

```yaml title="configs/bldg/multi_zone.yaml"
dataset: multizones_reference_buildings
building_type: OfficeSmall   # Warehouse, RetailStandalone, ...
split: train                 # train or test
index: 0                     # 0-based position in the split's ID list
```

### Running

```bash
# Default: train PPO on OfficeSmall
python scripts/train_multizones.py

# Switch building type
python scripts/train_multizones.py bldg.building_type=OfficeMedium

# Use SAC instead of PPO
python scripts/train_multizones.py policy=sac

# Train on the 3rd building of the OfficeMedium train split
python scripts/train_multizones.py \
    bldg.building_type=OfficeMedium bldg.split=train bldg.index=2

# Short run for debugging
python scripts/train_multizones.py \
    training.total_timesteps=10000 env.max_steps=2880
```

## Evaluation Protocol

1. **Train** on a pool of N buildings from a given type and split.
2. **Evaluate** the trained policy on every building in the pool (held-out
   episodes) as well as on the test split to measure within-type
   generalisation.
3. Report **per-building** and **aggregate** episode return, comfort
   violations, and energy consumption.

Combine with the [Cross-Environment Generalisation](cross-env.md) benchmark to
measure transfer across building types.
