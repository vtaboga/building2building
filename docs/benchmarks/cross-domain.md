# Cross-Domain Generalization

## Task Definition

Evaluate whether a policy trained on one building type can transfer to a
**different building type**. This tests the most extreme form of generalization:
the agent must handle completely different HVAC topologies, zone counts, and
thermal dynamics.

| Axis | Train | Test |
|---|---|---|
| Building type | Type A | Type B (different) |
| Reward | Fixed | Fixed |
| Action space | Fixed (by type) | Fixed (by type) |

## Metric

```python
import building2building as b2b

bench = b2b.benchmarks.CrossDomainGeneralization(difficulty="easy", task="task_const_e0")
scores = []
for env in bench.make_test_envs():
    traj = b2b.rollout(env, controller=my_policy)
    scores.append(
        b2b.compute_normalized_score(
            cumulative_return=float(traj.rewards.sum()),
            building_type=bench.test_type,
            task=bench.task,
            run_period="full_year",
            building_id=env.metadata["building_info"].building_id,
        )
    )
mean_score = sum(scores) / len(scores)
```

The score is `agent_return / baseline_return` on each test building; both
returns are negative, so **lower is better**: 1.0 matches the
reactive-controller baseline and a score below 1.0 beats it.
Cross-domain transfer is harder than dynamics adaptation
because the agent must generalise across HVAC topology (different action
dimension and observation structure).

## Settings

| Setting | Train Type | Test Type |
|---|---|---|
| `1` | `RetailStandalone` | `OfficeSmall` |
| `2` | `RetailStandalone` | `Warehouse` |
| `3` | `OfficeSmall` | `OfficeMedium` |

## API

```python
import building2building as b2b

bench = b2b.benchmarks.CrossDomainGeneralization(
    difficulty="easy",   # "easy", "medium", or "hard"
    task="task_const_e0",
    n_train=8,
    n_test=8,
)

# Access the building types
print(f"Train on: {bench.train_type}")
print(f"Test on: {bench.test_type}")

# Create environments
train_envs = bench.make_train_envs(n=4)
test_envs = bench.make_test_envs(n=4)
```
