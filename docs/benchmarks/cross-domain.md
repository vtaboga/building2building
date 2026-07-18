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

## Difficulty Levels

| Difficulty | Train Type | Test Type |
|---|---|---|
| `easy` | `RetailStandalone` | `OfficeSmall` |
| `medium` | `RetailStandalone` | `Warehouse` |
| `hard` | `OfficeSmall` | `OfficeMedium` |

The paper's Table 5 also defines a fourth setting (train on *n* building
types, test on *m* different types); it has no named difficulty preset —
compose it manually with `b2b.make_env` across types.

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

## Paper Experiments (Section 6.2)

The cross-domain benchmark uses the **Amorpheus** type-heterogeneous
transformer policy. Amorpheus leverages the morphology graph to apply
per-node-type encoders/decoders, enabling a single policy to operate across
buildings with different observation and action dimensions. In the paper's
experiment (the "n types → m types" setting), a single policy is trained
simultaneously on four building types (retail store, fast-food restaurant,
small office, medium office) and tested on unseen buildings from the test
split.

```bash
python -m baselines.train_cross_domain experiment=train_cross_domain
```

See [Baselines: Cross-Domain Transfer](../baselines/cross-domain.md) for
the full Amorpheus training pipeline.
