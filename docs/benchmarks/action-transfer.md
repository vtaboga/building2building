# Action-Space Transfer

## Task Definition

Evaluate whether a policy can adapt when the set of **controllable actuators
changes** between training and test. The building dynamics and reward stay
fixed; only the action interface changes. Train and test use the **same
building** — only the subset of actuators exposed to the agent differs.

| Axis | Train | Test |
|---|---|---|
| Building | Fixed | Fixed (same instance) |
| Reward | Fixed | Fixed |
| Actuator set | Reduced or full | Full or reduced |

## Metric

```python
import building2building as b2b

bench = b2b.benchmarks.ActionSpaceTransfer(
    system_type="unitary",
    direction="expand",
    task="task_const_e0",
)
traj = b2b.rollout(bench.make_test_env(), controller=my_policy)
score = b2b.compute_normalized_score(
    cumulative_return=float(traj.rewards.sum()),
    building_type=bench.building_type,
    task=bench.task,
    run_period="full_year",  # ActionSpaceTransfer always simulates the full year
    building_id=b2b.list_buildings(bench.building_type, bench.split)[bench.split_index],
)
```

The score is `agent_return / baseline_return` against the reactive-controller
baseline.  Both returns are negative, so **lower is better** — a score below
1.0 outperforms the reactive baseline.

## Settings

| System type | Training control | Test control | Action dim |
|---|---|---|---|
| Unitary | Air flow rate | Air flow + SAT | 5 → 10 |
| Central | VAV boxes only | VAV boxes + central SAT | 30 → 33 |
| Unitary | Air flow + SAT | Air flow rate | 10 → 5 |
| Central | VAV boxes + central SAT | VAV boxes only | 33 → 30 |

- **Unitary** (default: OfficeSmall, 5 zones) — the *reduced* action space
  removes supply-air temperature (SAT) setpoints, leaving only fan air
  mass flow rate.
- **Central** (default: OfficeMedium, 15 zones) — the *reduced* action
  space removes the per-loop central SAT actuator, leaving only VAV
  terminal actuators (damper + zone heating setpoint).

Excluded actuators are pinned at their default operating values (22 °C for
unitary SAT, 13 °C for central SAT) so the EnergyPlus simulation remains
physically valid.

## API

```python
import building2building as b2b

bench = b2b.benchmarks.ActionSpaceTransfer(
    system_type="unitary",   # "unitary" or "central"
    direction="expand",      # "expand" or "reduce"
    task="task_const_e0",
)

train_env = bench.make_train_env()
test_env = bench.make_test_env()

# With direction="expand", train has fewer actuators than test
print(f"Train action dim: {train_env.action_space.shape[0]}")
print(f"Test action dim:  {test_env.action_space.shape[0]}")
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_type` | `"unitary"` / `"central"` | `"unitary"` | HVAC system type. Determines which actuators are toggled and the default building type. |
| `direction` | `"expand"` / `"reduce"` | `"expand"` | Whether **test** has more or fewer actuators than training. |
| `task` | `str` | `"task_const_e0"` | Named task preset. |
| `building_type` | `str` or `None` | `None` | Override the default building type. When `None`, defaults to `"OfficeSmall"` for unitary and `"OfficeMedium"` for central. |
| `split` | `"train"` / `"test"` | `"train"` | Dataset split from which to select the building. |
| `split_index` | `int` | `0` | Index within the split to select the building. |

## Direction

- **`expand`**: training uses a reduced set of actuators; testing adds more
  actuators that the agent must learn to use.
- **`reduce`**: training uses the full actuator set; testing removes some,
  requiring the agent to maintain performance with fewer degrees of freedom.

## Actuator Subsets

### Unitary systems

Each zone is served by a dedicated HVAC unit with two actuators:

1. **Fan air mass flow rate** (always exposed)
2. **Supply air temperature setpoint** (removed in the reduced set, fixed at 22 °C)

### Central (VAV) systems

A central air loop serves multiple zones.  Per-terminal actuators are
always exposed; the per-loop actuator is toggled:

- **VAV terminal actuators** (always exposed): damper opening fraction +
  zone heating setpoint per zone.  Zone cooling setpoints are always
  fixed at 40 °C (standard B2B convention).
- **Central supply air temperature** (removed in the reduced set, fixed
  at 13 °C).
