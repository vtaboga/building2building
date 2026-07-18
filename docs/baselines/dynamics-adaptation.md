# Dynamics Adaptation Baselines

Baselines for the dynamics adaptation benchmark (Paper Section 6.1).

## Overview

Three training approaches are compared on the `DynamicsAdaptation` benchmark:

| Approach | Description | Wrappers |
|---|---|---|
| **Specialist** | Independent PPO per building | NormalizeObservation |
| **Baseline** | Single PPO across all train buildings | PadObservation + NormalizeObservation |
| **Parameterized** | Same + building parameter augmentation | PadObservation + AugmentObservation + NormalizeObservation |

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
| `hard` | `OfficeMedium` | 33 |

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
2. Wrap each in `PadObservation` to harmonize dimensions
3. (Parameterized only) Wrap in `AugmentObservationWithBuildingParams`
4. Wrap in `NormalizeObservation`
5. Use `ResampleBuildingOnResetWrapper` to sample a new building on each reset

## Evaluation

```bash
python -m baselines.eval_dynamics_adaptation \
    --model-path outputs/.../models/multi_parameterized.zip \
    --difficulty easy --approach parameterized
```

!!! warning "Known issue"

    `eval_dynamics_adaptation.py` hard-codes `PadObservation(env, target_size=20)`
    which may not match the training value. See [Known Issues](../about/known-issues.md).
