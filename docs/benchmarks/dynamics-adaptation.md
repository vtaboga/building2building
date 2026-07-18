# Dynamics Adaptation

## Task Definition

Evaluate whether a policy can generalize across **parametrically varied
buildings of the same type**. The reward and action space stay fixed; only the
building dynamics (insulation, orientation, HVAC sizing, etc.) change between
training and test.

| Axis | Train | Test |
|---|---|---|
| Building dynamics | Seen (train split) | Unseen (test split) |
| Reward | Fixed | Fixed |
| Action space | Fixed | Fixed |

## Metric

Evaluate with `compute_normalized_score` per test building, then average:

```python
import building2building as b2b

scores = []
for env in bench.make_test_envs(n=4):
    traj = b2b.rollout(env, controller=my_policy)
    scores.append(b2b.compute_normalized_score(traj))
mean_score = sum(scores) / len(scores)
```

A score of 0.0 matches the reactive-controller baseline; 1.0 is perfect.

## Difficulty Levels

| Difficulty | Building Type | Action Dim | Description |
|---|---|---|---|
| `easy` | `SingleFamilyHouse` | 2 | Single-zone residential |
| `medium` | `OfficeSmall` | 10 | 5-zone unitary office |
| `hard` | `OfficeMedium` | ~33 | Multi-zone VAV office |

## API

```python
import building2building as b2b

bench = b2b.benchmarks.DynamicsAdaptation(
    difficulty="easy",   # "easy", "medium", or "hard"
    task="task_const_e0",        # task preset
    n_train=900,         # number of training buildings
    n_test=71,           # number of test buildings
)

# Get building IDs
train_ids = bench.train_building_ids()  # list[str]
test_ids = bench.test_building_ids()    # list[str]

# Create environments
train_envs = bench.make_train_envs(n=4)
test_envs = bench.make_test_envs(n=4)
```

## Paper Experiments (Section 6.1)

Three training approaches are compared:

| Approach | Description | Script |
|---|---|---|
| **Specialist** | Independent PPO per building | `train_dynamics_specialist` |
| **Baseline** | Single PPO across all buildings (PadObs + NormObs) | `train_dynamics_baseline` |
| **Parameterized** | Same + building-parameter augmentation | `train_dynamics_parameterized` |

```bash
# Specialist
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_specialist difficulty=easy

# Parameterized (paper's main result)
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized difficulty=easy
```

See [Baselines: Dynamics Adaptation](../baselines/dynamics-adaptation.md) for
full details on reproducing these experiments.
