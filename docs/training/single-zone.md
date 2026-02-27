# Single-Zone Training

## Overview

Single-zone training is the simplest training setup in B2B: an SB3 agent is
trained on a single building with one conditioned zone.  This is the
recommended starting point for developing and debugging new algorithms before
scaling to multi-zone or multi-building settings.

## Script

```
scripts/train_single_zone_houses.py
```

The script uses `b2b.baselines.online_trainer.online_trainer()` as the
training backend and is configured via `configs/base.yaml`.

## Configuration

```yaml title="configs/base.yaml"
defaults:
  - _self_
  - wandb: default
  - training: default
  - policy: ppo
  - reward: deadband
  - task: default
  - bldg: single_family
  - rollout: default

env:
  normalize_obs: true
  normalize_action: false
  max_steps: null

n_episodes: 1
seed: 42

benchmark:
  split: train
  start: 0
  limit: 0
  max_steps: null
```

### Training Defaults

From `configs/training/default.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | 1,000,000 | Total training steps |
| `eval_freq` | 262,144 | Steps between evaluations |
| `eval_episodes` | 20 | Episodes per evaluation |
| `num_train_envs` | 4 | Parallel training environments |
| `cb_gradient_save_freq` | 500 | Gradient checkpoint frequency |

### Task Defaults

From `configs/task/default.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `run_period` | `full_year` | `full_year`, `winter`, or `summer` |
| `target_temperature_mode` | `constant` | `constant` or `occupancy` |
| `default_zone_target_temperature.occupied_c` | 21.0 | Occupied zone target (°C) |
| `default_zone_target_temperature.unoccupied_c` | 21.0 | Unoccupied zone target (°C) |

## Running

```bash
# Default: train PPO on a single-family house
python scripts/train_single_zone_houses.py

# Train with PPO explicitly
python scripts/train_single_zone_houses.py policy=ppo

# Train with SAC
python scripts/train_single_zone_houses.py policy=sac

# Shorter run for debugging
python scripts/train_single_zone_houses.py training.total_timesteps=50000

# Change reward function
python scripts/train_single_zone_houses.py reward=barrier

# Disable W&B logging
python scripts/train_single_zone_houses.py wandb.enabled=false

# Winter-only training
python scripts/train_single_zone_houses.py task.run_period=winter
```

## Output

Training output is written to the Hydra run directory.  Key artifacts:

| File | Description |
|---|---|
| `best_model.zip` | Best SB3 checkpoint (by evaluation return) |
| `final_model.zip` | Model at end of training |
| `evaluations.npz` | Evaluation returns at each `eval_freq` |
| `config.json` | Resolved Hydra config |
| `eplus_outputs/` | EnergyPlus simulation logs |

## SLURM

See [SLURM Execution](slurm.md) for cluster submission:

```bash
sbatch scripts/slurm/run_ppo.sh   # PPO training
sbatch scripts/slurm/run_sac.sh   # SAC training
```
