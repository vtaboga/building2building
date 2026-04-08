# Action-Space Transfer

## Task Definition

Evaluate whether a policy can adapt when the set of **controllable actuators
changes** between training and test. The building dynamics and reward stay
fixed; only the action interface changes.

This tests scenarios like:

- A building control system is upgraded with additional actuators
- Some actuators become unavailable due to maintenance
- The control interface is simplified or expanded

## API

```python
import building2building as b2b

bench = b2b.benchmarks.ActionSpaceTransfer(
    system_type="unitary",   # "unitary" or "central"
    direction="expand",      # "expand" or "reduce"
    task="task1",
    building_type="OfficeSmall",
    split_index=0,
)

train_env = bench.make_train_env()
test_env = bench.make_test_env()

# Compare action spaces
print(f"Train action dim: {train_env.action_space.shape[0]}")
print(f"Test action dim: {test_env.action_space.shape[0]}")
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_type` | `"unitary"` / `"central"` | `"unitary"` | HVAC system type |
| `direction` | `"expand"` / `"reduce"` | `"expand"` | Whether test has more or fewer actuators |
| `task` | `str` | `"task1"` | Named task preset |
| `building_type` | `str` | `"OfficeSmall"` | Building type |
| `split_index` | `int` | `0` | Building index |

## Direction

- **`expand`**: training uses a reduced set of actuators; testing adds more
  actuators that the agent must learn to use.
- **`reduce`**: training uses the full actuator set; testing removes some,
  requiring the agent to maintain performance with fewer degrees of freedom.
