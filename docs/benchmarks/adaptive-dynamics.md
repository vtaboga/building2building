# Adaptive Dynamics Benchmark

## Task Definition

Evaluate whether a controller can generalise across **parametrically varied
buildings of the same archetype**.  Unlike the cross-environment benchmark
(which varies building *type*), adaptive dynamics varies building *parameters*
— insulation levels, orientation, HVAC sizing, window-to-wall ratios, etc. —
within a single archetype (e.g. Hydro-Québec single-family houses).

This tests a policy's ability to adapt to the different thermal dynamics that
arise from construction variations in otherwise similar buildings.

## Key Class: `AdaptiveDynamicsProblem`

```python
from building2building.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem
```

`AdaptiveDynamicsProblem` is a frozen dataclass that encapsulates the benchmark
configuration and provides a `run()` method for executing rollouts:

```python
@dataclass(frozen=True, slots=True)
class AdaptiveDynamicsProblem:
    split: Literal["train", "test"] = "train"
    start: int = 0       # first building index to evaluate
    limit: int = 0       # max buildings (0 = all)
    max_steps: int | None = None
    base_config: dict[str, Any] | None = None
```

### Usage

```python
from building2building.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem

problem = AdaptiveDynamicsProblem(split="train", start=0, limit=0)
records = problem.run(policy, output_dir="./results")
```

Each `AdaptiveDynamicsRecord` returned contains the per-building rollout
results.

A convenience function is also available:

```python
from building2building.benchmark.problem_adaptive_dynamics import run

records = run(policy, split="train", output_dir="./results")
```

## Configuration

The script `scripts/bm_adaptive_dynamics.py` drives the benchmark using
`configs/bm_adaptive_dynamics.yaml`, which inherits from `configs/base.yaml`:

```yaml title="configs/bm_adaptive_dynamics.yaml"
defaults:
  - base
  - _self_
  - override /policy: unitary_g36

env:
  normalize_obs: false
  control_mode: hvac_actuators

hydra:
  run:
    dir: outputs/benchmarks/adaptive_dynamics/${now:%Y%m%d}/${now:%H%M%S}
```

The `benchmark` section in `configs/base.yaml` provides the split parameters:

```yaml title="configs/base.yaml (excerpt)"
benchmark:
  split: train
  start: 0
  limit: 0
  max_steps: null
```

### Running

```bash
# Evaluate on all train-split buildings
python scripts/bm_adaptive_dynamics.py benchmark.split=train benchmark.start=0 benchmark.limit=0

# Evaluate on test-split buildings
python scripts/bm_adaptive_dynamics.py benchmark.split=test

# Evaluate only the first 5 buildings
python scripts/bm_adaptive_dynamics.py benchmark.limit=5

# Use the ASHRAE air-loop controller instead of G36
python scripts/bm_adaptive_dynamics.py policy=ashrae_air_loop
```

## Policy Evaluation Across Building Variants

The benchmark iterates over every building in the selected split (or a subset
controlled by `start` and `limit`) and runs a full-episode rollout with the
provided policy.  Results include:

- **Per-building episode return** — measures control quality for each variant.
- **Aggregate statistics** — mean, standard deviation, min, and max across all
  building variants.
- **Energy consumption** — total and per-m² energy use.
- **Comfort metrics** — zone temperature violations relative to setpoints.

### Comparing Policies

Run the benchmark with different policies and compare aggregate performance:

```bash
# Baseline: G36 controller
python scripts/bm_adaptive_dynamics.py policy=unitary_g36

# Trained RL agent (requires a saved checkpoint)
python scripts/bm_adaptive_dynamics.py policy=sb3 policy.checkpoint_path=path/to/model.zip
```

## SLURM Execution

Dedicated SLURM scripts are provided for cluster execution:

```bash
sbatch scripts/slurm/run_rulebased_adaptive_dynamics.sh
sbatch scripts/slurm/run_ppo_baseline_adaptive_dynamics.sh
sbatch scripts/slurm/run_ppo_per_building_adaptive_dynamics.sh
sbatch scripts/slurm/run_ppo_parameterized_adaptive_dynamics.sh
```
