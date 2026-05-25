# Goal Adaptation

## Task Definition

Evaluate whether a policy can adapt when the **reward function or task changes**
while the building and action space stay fixed. This tests how well an agent
generalizes to different control objectives on the same physical system.

## API

```python
import building2building as b2b

bench = b2b.benchmarks.GoalAdaptation(
    building_type="OfficeSmall",
    split_index=0,
    train_task="task_occ_emed",   # train: occupancy setpoints, balanced trade-off
    test_task="task_occ_ehigh",   # test: same mode, higher energy emphasis
    run_period="full_year",
)

# Same building, different task
train_env = bench.make_train_env()
test_env = bench.make_test_env()

# Or create multiple copies
train_envs = bench.make_train_envs(n=4)
test_envs = bench.make_test_envs(n=4)
```

## Task Preset Pairs

Meaningful train/test combinations to explore:

| Train | Test | What Changes |
|---|---|---|
| `task_occ_emed` | `task_occ_e0` | Energy weight: balanced → comfort-only |
| `task_occ_emed` | `task_occ_ehigh` | Energy weight: balanced → energy-emphasis |
| `task_occ_emed` | `task_const_emed` | Setpoint mode: occupancy → constant |
| `task_occ_emed` | `task_rand_emed` | Setpoint mode: occupancy → random schedule |

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `building_type` | `BuildingType` | `"OfficeSmall"` | Building type to use |
| `split_index` | `int` | `0` | Index within the train split |
| `train_task` | `str` | `"task_occ_emed"` | Task preset for training |
| `test_task` | `str` | `"task_occ_ehigh"` | Task preset for testing |
| `run_period` | `str` | `"full_year"` | Simulation run period |
