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
    train_task="task1",   # train with Deadband, energy_weight=0.01
    test_task="task2",    # test with Deadband, energy_weight=0.10
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
| `task1` | `task2` | Energy weight increases 10x |
| `task1` | `task3` | Temperature mode: constant to occupancy |
| `task1` | `task4` | Reward type: deadband to barrier |
| `task3` | `task1` | Temperature mode: occupancy to constant |

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `building_type` | `BuildingType` | `"OfficeSmall"` | Building type to use |
| `split_index` | `int` | `0` | Index within the train split |
| `train_task` | `str` | `"task1"` | Task preset for training |
| `test_task` | `str` | `"task2"` | Task preset for testing |
| `run_period` | `str` | `"full_year"` | Simulation run period |
