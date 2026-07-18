# Goal Adaptation

## Task Definition

Evaluate whether a policy can adapt when the **reward function or task changes**
while the building and action space stay fixed. This tests how well an agent
generalises to different control objectives on the same physical system.

| Axis | Train | Test |
|---|---|---|
| Building | Fixed | Fixed (same instance) |
| Action space | Fixed | Fixed |
| Reward / task | Task A | Task B (different) |

The task family is the 6 normalized presets from the paper: 3 setpoint modes
(`const`, `occ`, `rand`) × 2 energy weights (`e0` for w_E = 0, `e05` for
w_E = 0.5), named `task_{const,occ,rand}_{e0,e05}`.

## Metric

```python
import building2building as b2b

bench = b2b.benchmarks.GoalAdaptation(
    train_task="task_occ_e05",
    test_task="task_occ_e0",
)
traj = b2b.rollout(bench.make_test_env(), controller=my_policy)
score = b2b.compute_normalized_score(
    cumulative_return=float(traj.rewards.sum()),
    building_type=bench.building_type,
    task=bench.test_task,
    run_period=bench.run_period,
    building_id=b2b.list_buildings(bench.building_type, "train")[bench.split_index],
)
```

The score is normalised against the reactive-controller baseline on the *test*
task: `agent_return / baseline_return`.  Both returns are negative, so **lower
is better** — a score below 1.0 beats the reactive controller.

## API

```python
import building2building as b2b

bench = b2b.benchmarks.GoalAdaptation(
    building_type="OfficeSmall",
    split_index=0,
    train_task="task_occ_e05",   # train: occupancy setpoints, energy priced in
    test_task="task_occ_e0",     # test: same mode, comfort-only
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
| `task_occ_e05` | `task_occ_e0` | Energy weight: 0.5 → 0 (comfort-only). The default "trade-off transfer" axis; both tasks are in the calibration regime. |
| `task_occ_e05` | `task_const_e05` | Setpoint mode: occupancy → constant |
| `task_occ_e05` | `task_rand_e05` | Setpoint mode: occupancy → random schedule |

The setpoint-mode-transfer pairs evaluate on tasks outside the calibration
regime, so env construction emits a one-time `RuntimeWarning` — this is
expected, not a bug.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `building_type` | `BuildingType` | `"OfficeSmall"` | Building type to use |
| `split_index` | `int` | `0` | Index within the train split |
| `train_task` | `str` | `"task_occ_e05"` | Task preset for training |
| `test_task` | `str` | `"task_occ_e0"` | Task preset for testing |
| `run_period` | `str` | `"full_year"` | Simulation run period |
