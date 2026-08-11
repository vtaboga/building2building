# Reproduce Dynamics Benchmark

This tutorial reproduces a mini version of the dynamics adaptation experiment
(paper Section 6.1) using the parameterized approach.

**Runnable script:** `tutorials/run_dynamics_benchmark.py`

---

## Overview

The dynamics adaptation benchmark trains a single policy across multiple
building instances of the same type, using `PadObservation` and
`AugmentObservationWithBuildingParams` to let the policy generalize across
varying dynamics.

---

## What the script covers

1. **Baseline returns** — the reactive-controller reference returns used by
   `compute_normalized_score` ship with the package
   (`building2building/scores/baseline_returns.csv`).  To regenerate them,
   run `python -m baselines.run_reactive_control experiment=eval_reactive_control`.

2. **Inspect the benchmark** — `b2b.benchmarks.DynamicsAdaptation(difficulty="easy",
   task="task_const_e0")` exposes `train_building_ids()` and
   `test_building_ids()`.

3. **Train with the CLI** (recommended):

    ```bash
    python -m baselines.train_dynamics_adaptation \
        experiment=train_dynamics_parameterized \
        difficulty=easy training.total_timesteps=50000
    ```

4. **Train programmatically** — wrap each training building with
   `PadObservation`, then `wrap_env_for_rl` (observation normalization +
   action rescaling), then `AugmentObservationWithBuildingParams`; use
   `ResampleBuildingOnResetWrapper` to cycle across buildings; train with
   SB3 `PPO`.

5. **Evaluate on test buildings** — for each test building ID, create an env,
   apply the same wrappers, and run one episode with `model.predict(...)`.

6. **Plot results** — `python -m baselines.plotting.plot_dynamics_adaptation
   --specialist-csv ... --baseline-csv ... --parameterized-csv ...`
   (CSVs produced by `baselines.eval_dynamics_adaptation`).

---

For the full programmatic pipeline, run:

```bash
python tutorials/run_dynamics_benchmark.py
```
