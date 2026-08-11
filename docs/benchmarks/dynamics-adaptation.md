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

bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task_const_e0")

scores = []
for building_id, env in zip(bench.test_building_ids(), bench.make_test_envs(n=4)):
    traj = b2b.rollout(env, controller=my_policy)
    scores.append(
        b2b.compute_normalized_score(
            cumulative_return=float(traj.rewards.sum()),
            building_type=bench.building_type,
            task=bench.task,
            run_period="full_year",
            building_id=building_id,
        )
    )
mean_score = sum(scores) / len(scores)
```

The score is `agent_return / baseline_return`. Both returns are negative
(cost-based reward), so **lower is better**: 1.0 matches the
reactive-controller baseline and a score below 1.0 beats it.

## Settings

| Setting | Building Type | Action Dim | Description |
|---|---|---|---|
| 1 | `SingleFamilyHouse` | 2 | Single-zone residential |
| 2 | `OfficeSmall` | 10 | 5-zone unitary office |
| 3 | `OfficeMedium` | 36 | Multi-zone VAV office |


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

