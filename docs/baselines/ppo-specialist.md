# PPO Specialist Training

Per-building PPO specialists (Paper Section 5, Appendix E.2).

## Overview

Each building type x task x building combination gets its own PPO policy
trained from scratch. This establishes the upper bound for what a single-building
specialist can achieve.

## Usage

```bash
# Full paper experiment
python -m baselines.train_ppo experiment=train_ppo

# Quick test: one building, short training
python -m baselines.train_ppo experiment=train_ppo \
    building_types=[OfficeSmall] tasks=[task_const_e0] \
    buildings_per_type=1 training.total_timesteps=100000

# Override hyperparameters
python -m baselines.train_ppo experiment=train_ppo \
    seed=42 policy.learning_rate=1e-4
```

## Configuration

The experiment config is at `baselines/configs/experiment/train_ppo.yaml`.

Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `building_types` | List of building types to train on | All 6 types |
| `tasks` | List of task presets | all 9 normalized presets |
| `buildings_per_type` | Number of buildings per type | `0` (all) |
| `training.total_timesteps` | PPO training steps per building | `1000000` |
| `training.n_envs` | Parallel training environments | `4` |

PPO hyperparameters follow the paper's Table 5 (see
`baselines/configs/policy/ppo.yaml`).

## Output

```
outputs/train_ppo/<date>/<time>/
├── models/
│   └── <building_type>/
│       └── <task>/
│           └── ppo_<building_id>.zip
└── results.csv
```

`results.csv` columns: `building_type`, `building_id`, `task`, `reward`,
`episode_length`, `normalized_score`.

## Evaluation

```bash
python -m baselines.eval_ppo --model-dir outputs/train_ppo/.../models
```

!!! warning "Known bug"

    `eval_ppo.py` expects flat model filenames but `train_ppo.py` saves in
    nested directories. See [Known Issues](../about/known-issues.md).

## Pipeline

1. For each (building_type, task, building_id):
   a. Create environment with `b2b.new_make_env()`
   b. Wrap with `NormalizeObservation`
   c. Train PPO for `total_timesteps`
   d. Evaluate for one episode
   e. Compute normalized score (requires `baseline_returns.csv`)
   f. Save model and append to results CSV
