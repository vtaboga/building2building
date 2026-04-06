# Cross-Environment Generalisation Benchmark

## Task Definition

Train on one set of building types and evaluate **zero-shot** on a disjoint
set of building types.  This benchmark measures how well an RL policy
generalises across fundamentally different building geometries, HVAC
topologies, and thermal dynamics.

## Key Classes

### `MultiTypeTrainTestBenchmark`

Use when train and test sets span **different building types**.

```python
from building2building.benchmark.problem_multizones_splits import MultiTypeTrainTestBenchmark
```

`MultiTypeTrainTestBenchmark` accepts separate lists of building types for the
train and test sides.  For each type it selects building IDs according to a
`SelectionSpec` (random, by index, or by metadata query) and builds the
corresponding `BuildingConfig` objects.

### `SingleTypeTrainTestBenchmark`

Use when train and test sets come from the **same building type** but different
building instances (within-type generalisation).

```python
from building2building.benchmark.problem_multizones_splits import SingleTypeTrainTestBenchmark
```

Both classes expose the same interface:

```python
benchmark.select_building_ids()   # -> (train_ids, test_ids)
benchmark.build_configs(eplus_output_dir=...)  # -> SplitBenchmarkResult
```

## Configuration

The benchmark is configured via `configs/benchmark_multizones_splits.yaml` and
the Hydra config group `configs/benchmark_interface/`.

### Multi-Type Example

```yaml title="configs/benchmark_interface/multi_type.yaml"
mode: multi_type

train:
  types: [OfficeSmall, Warehouse]
  selection:
    mode: random
    n: 2
    seed: 42

test:
  types: [OfficeMedium, RetailStandalone]
  selection:
    mode: random
    n: 1
    seed: 7
```

### Single-Type Example

```yaml title="configs/benchmark_interface/single_type.yaml"
mode: single_type
building_type: OfficeSmall

train:
  selection:
    mode: random
    n: 2
    seed: 42

test:
  selection:
    mode: random
    n: 2
    seed: 7
```

### Running

```bash
# Multi-type: train on OfficeSmall + Warehouse, test on OfficeMedium + RetailStandalone
python scripts/benchmark_multizones_splits.py benchmark_interface=multi_type \
  'benchmark_interface.train.types=[OfficeSmall,Warehouse]' \
  'benchmark_interface.test.types=[OfficeMedium,RetailStandalone]'

# Single-type: train/test within OfficeSmall
python scripts/benchmark_multizones_splits.py benchmark_interface=single_type \
  benchmark_interface.building_type=OfficeSmall
```

## Train / Test Side Configuration

Each side (train and test) can independently specify:

| Parameter | Description |
|---|---|
| `selection.mode` | `random`, `indices`, or `search_config` |
| `selection.n` | Number of buildings to select |
| `selection.seed` | Random seed for reproducibility |
| `config.reward` | Reward function (`BarrierRewardConfig`, `DeadbandRewardConfig`) |
| `config.task.run_period` | `winter`, `summer`, or `full_year` |
| `config.task.target_temperature_mode` | `constant` or `occupancy` |
| `config.actuator_access` | Control which actuators are available (see [Action-Space Shift](action-shift.md)) |

## Evaluation Protocol

1. **Select** train and test building IDs using seeded random sampling.
2. **Train** a policy on the train buildings (optionally using
   `ResampleBuildingOnResetWrapper` for multi-building episodes).
3. **Evaluate** the trained policy on the test buildings without any
   adaptation.
4. Report per-building and aggregate metrics: episode return, comfort
   violations, and energy consumption.

## Available Building Types

| Type | Description |
|---|---|
| `OfficeSmall` | Small commercial office |
| `OfficeMedium` | Medium commercial office |
| `Warehouse` | Commercial warehouse |
| `RetailStandalone` | Standalone retail building |
| `RestaurantFastFood` | Fast-food restaurant |
