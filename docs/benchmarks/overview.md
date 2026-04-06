# Benchmark Overview

## Motivation

Reinforcement learning for building control has shown promise, yet progress is
hampered by the lack of diverse, standardised evaluation protocols.  Most
existing benchmarks offer only a handful of buildings and a single evaluation
axis, making it difficult to study **transfer**, **generalisation**, and
**robustness** — the properties that matter most for real-world deployment.

Building2Building (B2B) fills this gap.  It provides a large collection of
EnergyPlus building models together with four complementary benchmark tasks,
each targeting a different dimension of policy generalisation.

## Benchmark Tasks

| Task | Question answered | Key class / wrapper |
|---|---|---|
| [Multi-Task Learning](multi-task.md) | Can a single policy learn to control *N* buildings at once? | `ResampleBuildingOnResetWrapper` |
| [Cross-Environment Generalisation](cross-env.md) | Does a policy trained on type-A buildings transfer to unseen type-B buildings? | `MultiTypeTrainTestBenchmark` |
| [Action-Space Shift](action-shift.md) | Can a policy adapt when the available actuators change at test time? | `ActuatorAccessConfig` |
| [Adaptive Dynamics](adaptive-dynamics.md) | Can a policy generalise across parametrically different buildings of the same archetype? | `AdaptiveDynamicsProblem` |

### Multi-Task Learning

Train a single policy on **N** buildings simultaneously.  At each episode
reset, `ResampleBuildingOnResetWrapper` samples a new building from the
training pool.  Wrappers such as `PadObservation` and
`AugmentObservationWithBuildingParams` harmonise the observation spaces across
heterogeneous buildings.

See [Multi-Task Learning](multi-task.md) for details.

### Cross-Environment Generalisation

Train on one set of building types (e.g. `OfficeSmall`, `Warehouse`) and
evaluate zero-shot on a disjoint set (e.g. `OfficeMedium`,
`RetailStandalone`).  The `MultiTypeTrainTestBenchmark` and
`SingleTypeTrainTestBenchmark` classes — defined in
`building2building.benchmark.problem_multizones_splits` — manage train/test building
selection and configuration.

See [Cross-Environment Generalisation](cross-env.md) for details.

### Action-Space Shift

Train with one actuator configuration and test with a different one.
`ActuatorAccessConfig` (from `building2building.config.models`) controls which actuators are
exposed.  For example, zone heating setpoints can be withheld during training
and restored at test time.

See [Action-Space Shift](action-shift.md) for details.

### Adaptive Dynamics

Evaluate a controller across parametrically varied buildings of the **same
archetype** (e.g. Hydro-Québec single-family houses with different insulation,
orientation, or HVAC sizing).  `AdaptiveDynamicsProblem` from
`building2building.benchmark.problem_adaptive_dynamics` orchestrates building iteration and
rollout.

See [Adaptive Dynamics](adaptive-dynamics.md) for details.

## Key Classes

```python
from building2building.benchmark.problem_multizones_splits import (
    SingleTypeTrainTestBenchmark,
    MultiTypeTrainTestBenchmark,
)
from building2building.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem
from building2building.config.models import ActuatorAccessConfig
from building2building.simulator.wrappers import (
    ResampleBuildingOnResetWrapper,
    PadObservation,
    AugmentObservationWithBuildingParams,
)
```

## Evaluation Principles

All benchmarks follow a common evaluation pattern:

1. **Reproducible splits** — building selections are seeded so that train/test
   sets are deterministic across runs.
2. **Configurable reward** — train and test sides can use independent reward
   functions (e.g. `BarrierRewardConfig` for training,
   `DeadbandRewardConfig` for evaluation).
3. **Season-aware tasks** — each side specifies a `run_period` (`winter`,
   `summer`, or `full_year`) and a target-temperature mode (`constant` or
   `occupancy`).
4. **Hydra configuration** — every benchmark is driven by composable YAML
   configs under `configs/`.
