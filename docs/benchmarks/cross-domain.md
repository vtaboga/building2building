# Cross-Domain Generalization

## Task Definition

Evaluate whether a policy trained on one building type can transfer to a
**different building type**. This tests the most extreme form of generalization:
the agent must handle completely different HVAC topologies, zone counts, and
thermal dynamics.

## Difficulty Levels

| Difficulty | Train Type | Test Type |
|---|---|---|
| `easy` | `RetailStandalone` | `OfficeSmall` |
| `medium` | `RetailStandalone` | `Warehouse` |
| `hard` | `OfficeSmall` | `OfficeMedium` |

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
buildings with different observation and action dimensions.

```bash
python -m baselines.train_cross_domain experiment=train_cross_domain
```

See [Baselines: Cross-Domain Transfer](../baselines/cross-domain.md) for
the full Amorpheus training pipeline.
