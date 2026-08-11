# Known Issues and Limitations

This page documents verified issues and limitations in the current codebase.
Issues from earlier reviews that have since been fixed (evaluation-script
argument order, model-path parsing, CSV schema mismatches, incomplete
`requirements.txt`, hard-coded observation padding) have been removed.

---

## Behaviour to be aware of

### Reward-normalizer calibration warnings

The seasonal energy normalizer `tau_E` in
`building2building/data/reward_normalizers.yaml` is calibrated with the
reference reactive controller in **occupancy mode with `dT = 1.0`**. Building
an env with a normalized task in `constant`/`random_schedule` mode, or with a
different `dT`, emits a one-shot `RuntimeWarning` from `create_simulator`.
This is **intentional** — it flags that the energy normalization constants were
calibrated under a different regime, not that anything is broken.

### `baseline_returns.csv` dependency

`compute_normalized_score()` in `building2building/scoring.py` reads the
packaged `building2building/scores/baseline_returns.csv`. If the file is
missing (e.g. a broken editable install), it raises `FileNotFoundError` with
regeneration instructions. Tuples of
`(building_type, task, run_period, building_id)` that are not in the packaged
evaluation grid raise a `KeyError` — you must regenerate the CSV (see
`baselines/run_reactive_control.py`) before scoring outside that grid.

### EnergyPlus native memory growth

EnergyPlus retains roughly 14 MB of RSS per create/reset/close cycle that is
not released back to the OS (measured in `tests/long/test_env_leak.py`).
Long-running jobs that create many envs in one process should periodically
recycle the worker process — `tests/long/test_all_buildings_env_smoke.py`
batches buildings into fresh subprocesses for exactly this reason.

### Some experiment configs fail to compose

The `baselines/configs/experiment/train_dynamics_*.yaml` (all three) and
`train_cross_domain.yaml` configs currently contain
`override /reward: task_const_e0`, but no `reward/` config group exists, so
`python -m baselines.train_dynamics_adaptation experiment=train_dynamics_baseline`
(and the `specialist`/`parameterized`/`train_cross_domain` variants) fail at
Hydra compose time with `Could not override 'reward'`. A fix updating these
configs to the current task-preset mechanism is in progress. Until it lands,
remove the stale `override /reward:` line from the experiment config.

### Hydra `@package _global_` merging

Experiment configs in `baselines/configs/` use `@package _global_`, which
merges keys into the top-level config. Some experiment configs define
`training:` overrides that merge with (not replace) the defaults. This is
correct Hydra behavior but can be confusing.

---

## Test Coverage Gaps

| Area | Coverage |
|---|---|
| `building2building/` (core) | Good -- quick tests cover API, registry, scoring, types, benchmarks, config, wrappers, morphology, pipeline |
| `baselines/` (scripts) | Smoke tests only (`test_train_ppo_smoke.py`, `test_train_sac_smoke.py`, `test_run_reactive_control_smoke.py`, `test_eval_path_layout.py`) -- the Hydra/CLI entry points are never invoked with real configs |
| Integration pipeline | **No test** for the full train -> eval -> plot pipeline |
| Hydra config | **No test** for config composition (experiment + policy + reward) |
| Long tests | Require EnergyPlus and network access (HuggingFace) |

---

## Roadmap / TODO

- [ ] Invoke the baselines training/evaluation entry points with real Hydra configs in tests
- [ ] Add an integration test for the full experiment pipeline
- [ ] Add Hydra config composition tests
