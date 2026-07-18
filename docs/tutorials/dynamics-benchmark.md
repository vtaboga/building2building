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

1. **Generate baseline returns** (optional pre-step) — run
   `python -m baselines.run_reactive_control` on the relevant buildings so
   that `compute_normalized_score` has reference values.

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
   `PadObservation` + `AugmentObservationWithBuildingParams` +
   `NormalizeObservation`; use `ResampleBuildingOnResetWrapper` to cycle
   across buildings; train with SB3 `PPO`.

5. **Evaluate on test buildings** — for each test building ID, create an env,
   apply the same wrappers, and run one episode with `model.predict(...)`.

6. **Plot results** — `python -m baselines.plotting.plot_dynamics_adaptation`.

---

For the full programmatic pipeline, run:

```bash
python tutorials/run_dynamics_benchmark.py
```
