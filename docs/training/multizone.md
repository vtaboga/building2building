# Multizone Training

## Overview

Multizone training targets commercial buildings with multiple conditioned
zones.  B2B provides a collection of reference buildings across six commercial
types, each with pre-generated train/test splits.  A single agent is trained on
a specific building instance (selected by type, split, and index).

## Script

```
scripts/train_multizones.py
```

The script calls `b2b.training.run_multizones_training()` and is configured
via `configs/train_multizones.yaml`.

## Configuration

```yaml title="configs/train_multizones.yaml"
defaults:
  - _self_
  - wandb: default
  - training: default
  - policy: ppo
  - reward: deadband
  - task: default

multizones:
  building_type: OfficeSmall   # Warehouse, HotelSmall, RetailStandalone,
                               # RestaurantFastFood, OfficeMedium, OfficeSmall
  split: train                 # train or test
  index: 0                     # 0-based position in the split's ID list

env:
  normalize_obs: true
  normalize_action: false
  max_steps: null              # defaults to task.run_period horizon

seed: 42
```

### Building Types

| Type | Description |
|---|---|
| `OfficeSmall` | Small commercial office |
| `OfficeMedium` | Medium commercial office |
| `Warehouse` | Commercial warehouse |
| `HotelSmall` | Small hotel |
| `RetailStandalone` | Standalone retail building |
| `RestaurantFastFood` | Fast-food restaurant |

## Running

```bash
# Default: train PPO on OfficeSmall (train split, index 0)
python scripts/train_multizones.py

# Switch building type
python scripts/train_multizones.py multizones.building_type=OfficeMedium

# Train on a specific building instance
python scripts/train_multizones.py \
    multizones.building_type=OfficeMedium \
    multizones.split=train \
    multizones.index=2

# Use SAC instead of PPO
python scripts/train_multizones.py policy=sac

# Short run for debugging
python scripts/train_multizones.py \
    training.total_timesteps=10000 env.max_steps=960

# Winter-only training
python scripts/train_multizones.py task.run_period=winter

# Occupancy-based target temperature
python scripts/train_multizones.py task.target_temperature_mode=occupancy
```

## Output Structure

Hydra output is organised by building type, split, algorithm, and timestamp:

```
outputs/train_multizones/<building_type>/<split>_<index>/<algorithm>/<timestamp>/
├── best_model.zip
├── final_model.zip
├── evaluations.npz
├── config.json
└── eplus_outputs/
```

## Multi-Building Training

To train a single policy across **multiple** buildings simultaneously, use the
[Multi-Task Learning](../benchmarks/multi-task.md) benchmark setup with
`ResampleBuildingOnResetWrapper`.  The `scripts/train_multizones.py` script
trains on a single building instance per run.

## Scaling Tips

- **Parallelise across buildings** — submit one SLURM job per
  `(building_type, split, index)` combination.
- **Shared evaluation** — use the benchmark scripts to evaluate all trained
  checkpoints on a common test set.
- **Observation normalisation** — `env.normalize_obs=true` (the default) is
  recommended for RL training on multizone buildings, as zone counts and
  observation ranges vary across types.
