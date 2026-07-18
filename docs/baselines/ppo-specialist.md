# PPO & SAC Specialist Training

Per-building PPO and SAC specialists (Paper Section 5, Appendix E.2).

## Overview

Each building type × task × building combination gets its own policy trained
from scratch. This establishes the upper bound for what a single-building
specialist can achieve. Two algorithms are provided — on-policy **PPO**
(`train_ppo.py`) and off-policy **SAC** (`train_sac.py`) — sharing the same
Hydra interface, building loop, and `b2b.make_env` / wrapper stack.

## Usage

```bash
# Full paper experiment
python -m baselines.train_ppo experiment=train_ppo
python -m baselines.train_sac experiment=train_sac

# Quick test: one building, short training
python -m baselines.train_ppo experiment=train_ppo \
    building_types=[OfficeSmall] tasks=[task_const_e0] \
    buildings_per_type=1 training.total_timesteps=100000

# Override hyperparameters
python -m baselines.train_ppo experiment=train_ppo \
    seed=42 policy.learning_rate=1e-4

# Task-study sweeps on the fast (test_small) split, single seed
python -m baselines.train_ppo experiment=train_ppo_task_study \
    --multirun seed=0 split=test_small
python -m baselines.train_sac experiment=train_sac_task_study \
    --multirun seed=0 split=test_small
```

## Configuration

The experiment configs are at `baselines/configs/experiment/train_ppo.yaml`
and `train_sac.yaml`.

Key parameters:

| Parameter | Description | PPO default | SAC default |
|---|---|---|---|
| `building_types` | Building types to train on | All 6 types | All 6 types |
| `tasks` | List of task presets | all 6 presets (`task_{const,occ,rand}_{e0,e05}`) | `task_{const,occ,rand}_e0` |
| `split` | Dataset split sampled from | `test` | `test` |
| `buildings_per_type` | Buildings per type | `8` | `8` |
| `run_period` | Simulation period | `full_year` | `full_year` |
| `training.total_timesteps` | Training steps per building | `5_000_000` | `2_000_000` |
| `training.n_envs` | Parallel training environments | `14` | `4` |

PPO hyperparameters follow the paper's Table 8 (`baselines/configs/policy/ppo.yaml`);
SAC uses `baselines/configs/policy/sac.yaml` with `baselines/configs/training/sac.yaml`.

## Output

Outputs land under `${base_dir}/b2b/train_ppo/<date>/<time>/`, where
`base_dir` defaults to `$SCRATCH` (or `./outputs` when unset):

```
<output_dir>/
├── models/
│   └── <building_type>/
│       └── <task>/
│           └── ppo_<building_id>.zip
├── tensorboard/
└── results.csv
```

`results.csv` columns: `building_type`, `building_id`, `task`,
`total_reward`, `normalized_score`.

## Evaluation

PPO models can be re-evaluated from disk with `eval_ppo.py`:

```bash
python -m baselines.eval_ppo --model-dir outputs/train_ppo/.../models
```

`eval_ppo.py` recursively discovers models saved as
`models/<building_type>/<task>/ppo_<building_id>.zip`, re-evaluates each on
its building/task, and writes `<base_dir>/b2b/eval/ppo_results.csv`
(override with `--output`). Additional flags: `--n-episodes` and
`--run-period {full_year,winter,summer}`.

`train_sac.py` runs a single post-training evaluation episode and writes its
result directly to the results CSV, so there is no separate `eval_sac` step.

## Pipeline

1. For each (building_type, task, building_id):
   a. Create environment with `b2b.make_env()`
   b. Wrap for RL: deterministic `[0, 1]` observation scaling
      (`NormalizeObservation`) plus `[-1, 1]` action rescaling
   c. Train PPO or SAC for `total_timesteps`
   d. Evaluate for one episode
   e. Compute normalized score (requires `baseline_returns.csv`)
   f. Save model and append to results CSV
