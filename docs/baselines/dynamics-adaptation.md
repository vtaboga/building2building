# Dynamics Adaptation Baselines

Baselines for the dynamics adaptation benchmark (Paper Section 6.1).

## Overview

Three training approaches are compared on the `DynamicsAdaptation` benchmark:

| Approach | Description | Wrappers |
|---|---|---|
| **Specialist** | Independent PPO per building | none (raw observations) |
| **Baseline** | Single PPO across all train buildings | PadObservation + NormalizeObservation |
| **Parameterized** | Same + building parameter augmentation | PadObservation + NormalizeObservation + AugmentObservationWithBuildingParams |

## Usage

### Specialist

```bash
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_specialist difficulty=easy
```

### Multi-Building Baseline

```bash
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_baseline difficulty=easy
```

### Parameterized (paper's main result)

```bash
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized difficulty=easy
```

## Difficulty Levels

| Level | Building Type | Action Dim |
|---|---|---|
| `easy` | `SingleFamilyHouse` | 2 |
| `medium` | `OfficeSmall` | 10 |
| `hard` | `OfficeMedium` | 36 |

## Configuration

Each approach has its own experiment config:

- `baselines/configs/experiment/train_dynamics_specialist.yaml`
- `baselines/configs/experiment/train_dynamics_baseline.yaml`
- `baselines/configs/experiment/train_dynamics_parameterized.yaml`

Key overrides:

```bash
# Change difficulty
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized difficulty=medium

# Short training for testing
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized difficulty=easy \
    training.total_timesteps=10000
```

## Multi-Building Training

The baseline and parameterized approaches use `ResampleBuildingOnResetWrapper`
to train across multiple buildings simultaneously:

1. Build a pool of training buildings from the benchmark
2. Wrap the pool in `ResampleBuildingOnResetWrapper` so a new building is
   sampled on each reset
3. Wrap in `PadObservation` to harmonize dimensions — the padding size is
   auto-detected by probing a few training buildings for the maximum
   observation dimension, and recorded in a `metadata.json` in the run's
   output directory
4. Wrap for RL: deterministic `[0, 1]` observation scaling
   (`NormalizeObservation`) plus `[-1, 1]` action rescaling
5. (Parameterized only) Wrap in `AugmentObservationWithBuildingParams`

## Evaluation

```bash
python -m baselines.eval_dynamics_adaptation \
    --model-path outputs/.../models/multi_parameterized.zip \
    --difficulty easy --approach parameterized --task task_const_e0
```

For the multi-building approaches, the observation padding size used during
training is read from a `metadata.json` looked up next to the model file;
pass `--pad-obs-size` explicitly if it is not found there (training writes
`metadata.json` at the run's output-dir root, one level above `models/`).
Specialist models are evaluated
by pointing `--model-path` at the directory containing the
`specialist_<building_id>.zip` files. Results are written to
`<base_dir>/b2b/eval/dynamics_results.csv` (override with `--output`).
