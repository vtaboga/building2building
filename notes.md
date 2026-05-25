<!-- -*- mode: markdown -*- -->

# Notes

Compact, in-flight working notes. Read this **before** starting any task.
Update as the project moves; keep it short and context-efficient.

Companion: `design_doc.md` (slow-changing intent + roadmap),
`TODO.md` (atomic action items), `paper/main.tex` (final results).

Distilled from `summaries/` (which will be deleted once everyone agrees
this file captures what is still relevant).

---

---

## Calibration sanity plot

The calibration sanity figure lives **outside the git-tracked tree**
(inside the git-ignored `/analysis/` directory):

```
analysis/task_study/reward_design/plots/fig_random_policy_normalizer_calibration.png
```

It was produced by the `--mode aggregate` step of
`compute_random_policy_reward_normalizers.py` at git sha
`00e7c24aa12e3b27c19b78e813bca64bef390c43` (the same run that wrote
`reward_normalizers.yaml`).  Per-bucket median of
`temp_penalty/τ_T` and `power_penalty/τ_E` is exactly 1.0 by
construction (tau values store the median), and bucket IQRs match the
`tau_T_iqr` / `tau_E_iqr` fields in the committed YAML.

To regenerate:

```bash
python -m analysis.task_study.compute_random_policy_reward_normalizers --mode aggregate
```

(Requires the per-building rollout cache under
`$SCRATCH/b2b_reward_normalizers_random/data/`.)

---

## Reward — calibration regime and impl map

Calibration regime (load-bearing for Phase A/B/C):

- Calibration task: `task_occ_*` family (occupancy-driven setpoints,
  seasonal unoccupied: winter 18 °C / shoulder 21 °C / summer 26 °C),
  `dT = 1.0`.
- Run period: full year by default; YAML supports per-season slicing
  (`winter`, `summer`, `full_year`).
- Split: train.
- Aggregation: median across buildings within each
  `(building_type, climate_zone)` bucket. `SingleFamilyHouse` is one
  bucket (key `cz0`, no ASHRAE CZ assignment).

Task family: nine presets `task_<mode>_<level>` over the 3×3 grid.
mode ∈ {`const`, `occ`, `rand`}; level ∈ {`e0`, `emed`, `ehigh`}
with `w_E ∈ {0.0, 1.0, 5.0}`.

Implementation map:

| Concern | Location |
| --- | --- |
| Reward config dataclass (sentinel pattern) | `building2building/types.py::NormalizedDeadbandRewardConfig` |
| Reward function | `building2building/simulator/rewards.py::NormalizedDeadbandReward` |
| Calibration constants loader | `building2building/data/reward_normalizers.py` |
| Calibration YAML | `building2building/data/reward_normalizers.yaml` |
| Task presets + factory | `building2building/config/tasks.py` |
| Env-factory auto-fill of `(τ_T, τ_E)` | `building2building/api/__init__.py::new_make_env` |
| Calibration scripts | `analysis/task_study/compute_reward_normalizers.py`, `compute_random_policy_reward_normalizers.py` |

---

## Known training pathologies under the new reward

These are documented mechanisms, not bugs. They drive Phase B.

### PPO

- Freezes after the first gradient update at `w_E ∈ {10, 50, 100}` on
  `winter`. At `w_E=1`, 1 of 6 buildings freezes; at `w_E=100`, 6 of 6.
- Mechanism: initial gradient dominated by the power axis →
  saturates an action channel → power gradient collapses → temperature
  gradient swamped by value-function noise → `target_kl=0.02` (very
  conservative) prevents recovery.
- Practical safe range on winter: `w_E ∈ {0, 1, 5}`.
- Findings are **single-seed (seed 0), winter only, 6 buildings**. Has
  not been reproduced on `full_year` (the calibration regime).
- First knobs to try if rescuing high-`w_E` learning matters:
  `target_kl → None`, `log_std_init: -1.0 → 0.0`, `ent_coef: 0.01 → 0.05`.

### SAC

- Chatters (heavy bang-bang oscillation), drifts toward action
  boundaries during training (action saturation rises monotonically
  from 0% → 25% over 300 k steps), and degrades on deterministic eval
  after an early peak. 12/18 runs in the existing sweep show
  best > final by > 0.1.
- Root causes ranked by impact:
  1. `log_std_init = -3.0` is too low (initial post-tanh actions live
     in [-0.05, 0.05]; agent commits before exploring).
  2. `ent_coef: auto` mis-tunes against post-tanh effective entropy
     (which is small once actions saturate); auto-α decreases.
  3. gSDE noise is heavily correlated and reinforces boundary-side
     pushing rather than relieving it.
- Recommended fixes (live in `baselines/configs/policy/sac.yaml`):
  Fix 1: `log_std_init: 0.0`. Fix 2: `ent_coef: 0.2`.
  Fix 3: `use_sde: false`. Fix 4 (tertiary): `tau: 0.005`,
  `gradient_steps: 1`, `train_freq: 4`. Fix 5 (tooling): save model
  snapshots at every eval interval in the main study.
- Literature concurs that occupancy-driven setpoints + 5-min control
  cadence are intrinsically hard for SAC. Mitigations worth keeping
  in mind if the cheap fixes are insufficient: forward-looking
  schedule features in the obs, `||a_t - a_{t-1}||²` action-rate
  penalty, repeat-action wrapper / lower control cadence, observation
  history.

### SAC B2 ablation — 5-cell × 3-seed × 6-building on task_occ_emed / full_year

Script: `analysis/task_study/sac_diagnostic/submit_ablation.sh`
Submit: `sbatch --array=0-89 analysis/task_study/sac_diagnostic/submit_ablation.sh`
Data: `$SCRATCH/b2b_sac_diagnostic/cell{N}_{name}/seed{S}/data/`
Status: **PENDING** (not yet submitted)

Ablation cells (all other hparams from the B1 sac.yaml):

| Cell | Name | log_std_init | ent_coef | use_sde |
|------|------|-------------|----------|---------|
| 0 | baseline   | −3.0 | auto | True  |
| 1 | fix_log_std | 0.0 | auto | True  |
| 2 | fix_ent_coef | −3.0 | 0.2 | True  |
| 3 | fix_sde | −3.0 | auto | False |
| 4 | all_fixes  |  0.0 | 0.2 | False |

Acceptance: cell 4 shows 0/18 runs with `best − final > 0.1`;
action saturation < 10% at every checkpoint.

Results (to be filled after run completes):

<!-- RESULT PLACEHOLDER — replace once sbatch finishes -->
| Cell | % degraded (best−final>0.1) | action sat. | notes |
|------|----------------------------|-------------|-------|
| 0    | _/18                       | _           |       |
| 1    | _/18                       | _           |       |
| 2    | _/18                       | _           |       |
| 3    | _/18                       | _           |       |
| 4    | _/18                       | _           |       |

### Cross-checks under PPO/SAC reruns

- Constant-setpoint sweep (`task_const_*`) is a useful confounder
  control: if constant-target SAC is materially more stable than
  occupancy SAC at the same `w_E`, then occupancy is the
  destabiliser, not the reward.
- The PPO obs/action normalization landed but **no sweep has been
  re-run with it**. Old `.npz` checkpoints under
  `$SCRATCH/b2b_reward_design_normalized/data/` are not comparable
  to new ones (the per-step reward distributions differ at every
  point). Delete relevant `.npz` files before any new comparison sweep.

---

## Reward family — implementation gotchas

- `NormalizedDeadbandRewardConfig` has **two valid states**: unfilled
  sentinel (`tau_T = tau_E = None`, used by presets) and filled
  (both floats > 0, ready for the simulator). Mixed state is always
  a bug — the dataclass `__post_init__` raises.
- `lru_cache` on `_cached_load` (capacity 12, keyed by path +
  run_period). Tests that mutate the YAML must call
  `clear_reward_normalizers_cache()`.
- Reward decomposition for analysis: do not reuse the legacy
  `temp = -reward - w * power` formula on normalized runs. Use
  both `tau_T` and `tau_E`. Checkpoint `.npz` files written by
  `_eval_and_save` carry the tau values for this reason.
- `task_occ_*` is the on-calibration regime by construction.
  `task_const_*` and `task_rand_*` reuse the same `(τ_T, τ_E)` (these
  are calibration constants per `(bt, cz)`, not per task), so off-mode
  usage is approximate; the simulator emits a `RuntimeWarning` (deduped
  per `(bt, bid, mode, dT)`).
- Tail buildings drift from `w_E=1` semantics: at the 25th/75th
  percentile of a loose bucket the effective `w_E` can be off by
  30–60%. Cross-bucket comparisons should report bucket and
  percentile when possible.
- YAML coverage: 5393/5400 train buildings; 7 SFH simulation crashes
  (currently absorbed by the median over the remaining 893).

---

## RL env wrapper stack

All RL training and eval (PPO, SAC, dynamics adaptation, cross-domain)
go through `make_rl_env_fn` (`baselines/utils/training.py`). The wrapper
stack is exactly:

```
Monitor ( NormalizeObservation ( RescaleAction ( TimeLimit ( EnergyPlusSimulator ) ) ) )
```

- `NormalizeObservation` is **deterministic** (`[0, 1]` bounds from
  `simulator/observation_spaces.py::flat_observation_info`). No
  `VecNormalize` stats file. The wrapper does **not** clip; values
  outside `[0, 1]` during warmup are expected.
- `RescaleAction(-1, 1)` is applied **inside** `new_make_env` (with
  `rescale_action=True`); `wrap_env_for_rl` is then called with
  `rescale_action=False` to avoid double-rescaling.
- Reactive controllers, tuning scripts, and `train_cross_domain.py`
  (Amorpheus) bypass this stack and use raw engineering units.
- `train_dynamics_adaptation.py::train_specialist` is the one
  exception that does not currently apply `NormalizeObservation`. If
  Phase C reruns the specialist path, add an explicit
  `wrap_env_for_rl` call there.

---

## Bugs to fix during Phase D

(Documented in the README "Known Issues" section as of 2026-05.)

- `compute_normalized_score` argument order swapped in
  `baselines/eval_ppo.py` and `baselines/eval_dynamics_adaptation.py`.
- `eval_ppo.py` model path mismatch: `train_ppo.py` saves to
  `models/<type>/<task>/ppo_<id>.zip`; `eval_ppo.py` looks for
  `ppo_<type>_<id>_<task>.zip`.
- `plot_ppo_specialist.py` expects `reward_mean` column;
  `eval_ppo.py` writes `reward`.
- `eval_dynamics_adaptation.py` hardcodes
  `PadObservation(env, target_size=20)`; must be read from training
  metadata.
- `baselines/requirements.txt` is incomplete (missing `hydra-core`,
  `omegaconf`, `optuna`, `matplotlib`, `pyyaml`); duplicates
  `pyproject.toml[training]`. Drop it; point users at
  `pip install -e ".[training]"`.

---

## Operational gotchas (cluster / SLURM)

- `sbatch --wrap='... source .venv/bin/activate ...'` runs under
  `/bin/sh` which has no `source`. Use a real `.sh` launcher with
  `#!/bin/bash`.
- `_eval_and_save` skips rewriting existing `.npz` files. Changing
  the eval cadence retroactively produces a Frankenstein grid; delete
  the old files first.
- `SubprocVecEnv` workers are single-threaded EnergyPlus; `N_ENVS =
  cpus - 2` is the sweet spot. Default sweeps use 14 envs on 16 CPUs.
- `task_occ_e0` has `w_E = 0`, so the dominance-ratio plot is NaN
  by construction — expected, not a bug.
- **EnergyPlus resource leak (B0) — fixed upstream.** The leak-free
  lifecycle (stop the simulation thread, join it, `gc.collect()`, rmtree
  the output dir) lives on `minergym.environment.EnergyPlusEnvironment`
  itself. `ManagedState.finalize` → `delete_state` fires correctly after
  every `close()`. `close_env_aggressively` is now a thin
  `DeprecationWarning` shim; all callers use plain `env.close()`.
  Tests: `tests/long/test_env_leak.py` (filesystem cleanup, thread join,
  regression RSS guard, plain-close regression).

  **Upstream commits on vtaboga/minergym:**
  - `6d03b9a` — initial `close()` with thread join + gc + optional
    `rmtree`; `reset()` delegates to `close()`; new constructor
    parameters `eplus_output_dir`, `cleanup_output_dir_on_close`,
    `thread_join_timeout`.
  - `956c3e1` — gate rmtree on `had_ep` (idempotency fix) and add
    `eplus_output_dir.mkdir()` in `reset()` so the directory exists for
    every `make_energyplus()` call regardless of
    `cleanup_output_dir_on_close`.

  `pyproject.toml` is pinned to `956c3e1`. `building2building/simulator`
  no longer subclasses `EnergyPlusEnvironment`: `create_simulator()`
  constructs it directly with `cleanup_output_dir_on_close=eplus_output_dir
  is not None`.

  **Residual EnergyPlus-native RSS growth (~14 MB/cycle, irreducible).**
  Even with `delete_state` and `reset_state`, EnergyPlus accumulates
  ~14 MB/cycle in C++ global/static objects inside the DLL (output variable
  registries, HVAC manager tables, etc.) that live outside the
  `EnergyPlusData` state object and cannot be freed from Python.
  Confirmed via `/proc/smaps` (growth is in `[heap]`/`[anon]`) and
  `tracemalloc` (Python allocations stable after cycle 1).  The only
  complete fixes are (a) an upstream EnergyPlus refactor to move those
  globals into `EnergyPlusData`, or (b) subprocess isolation (run each
  simulation in a subprocess so the OS reclaims everything on exit).
  At ~14 MB/cycle, a 500-episode SLURM job accumulates ~7 GB; stay within
  node memory budget when planning long training runs.

  **B0.1.f (housekeeping):** `gitpython>=3.1.0` was bundled into commit `be7463c`
  alongside the B0 resource-leak fix; it belongs to `building2building/store.py`
  and is unrelated to B0.  The dependency is correct and the behaviour is
  unaffected; this note preserves the audit trail.

  **B0/B0.1 complete as of commits `5118af5`–`7dd389f` (in-tree) and
  `956c3e1` (upstream).** Both `close()` and `reset()` are now leak-free
  by default.  The `had_ep` guard in the upstream `close()` is
  load-bearing: it makes `close()` idempotent so the polymorphic call
  from `reset()` (which calls `self.close()` after the prior episode has
  already been stopped) does not re-rmtree the freshly recreated output
  directory.

  **What to watch for in future experiments** (issues that could appear but
  are not covered by the automated tests):

  1. **"EnergyPlus thread did not exit within Xs" in `.err` logs.**  This
     warning fires from `close()` when the thread is still alive after
     `thread_join_timeout` (default 10 s).  It means EnergyPlus is stuck in
     a C-level call and the thread leaked.  Check: large buildings with many
     warmup phases, heavily loaded nodes with CPU contention, or simulations
     killed mid-step.  Mitigation: increase `thread_join_timeout` in
     `new_make_env`; long-term fix is subprocess isolation.

  2. **`$SLURM_TMPDIR` growing unexpectedly during training.**  The output dir
     is rmtree'd inside `close()` only when `had_simulation` is `True`.  If a
     job crashes between `make_energyplus()` and the first `reset()` (i.e. the
     env is constructed but never started), the dir is never cleaned up.
     Similarly, if `env.close()` is never called (KeyboardInterrupt, SIGKILL),
     the dir leaks.  Check `$SLURM_TMPDIR` usage at the end of a failed job;
     if it is large, the crash was mid-simulation rather than post-cleanup.

  3. **RSS growing faster than ~14 MB/episode.**  The known residual is
     ~14 MB/cycle (EnergyPlus C++ globals).  If W&B or SLURM job stats show
     RSS growing at >20 MB/episode, the Python-side fix is likely regressed
     (e.g. `self.ep = None` is missing, `gc.collect()` was removed, or a
     wrapper is holding a reference to the old env).  Run
     `tests/long/test_env_leak.py` on a node with EnergyPlus to isolate.

  4. **Thread count creeping up across Optuna trials in a single process.**
     Optuna runs multiple trials sequentially in one process.  Each trial
     calls `new_make_env` → `reset()` → ... → `close()`.  If the thread
     count after a trial is `baseline + 1`, the join timed out for that trial
     (see point 1).  Watch `threading.active_count()` in the Optuna callback
     or add a `gc.collect()` + assertion after each trial.

  5. **`reset()` returning stale observations after a long-running episode.**
     The upstream `reset()` calls `self.close()` before `make_energyplus()`.
     If `try_stop()` raises (e.g. because EnergyPlus crashed during the
     episode), the exception will propagate.  This is the correct "fail
     loudly" behaviour, but watch for unexpected `reset()` failures in
     W&B run logs — they indicate an EnergyPlus crash that was previously
     silently swallowed.

  6. **Output-dir accumulation with `ResampleBuildingOnResetWrapper`.**  The
     wrapper calls `self.env.close()` only when the building index changes.
     When the same building is resampled, the inner `env.reset()` is called
     (the upstream leak-free reset), which cleans up the old dir and recreates
     it.  If the wrapper is ever bypassed or subclassed differently, verify
     that the inner env's `reset()` is the upstream
     `EnergyPlusEnvironment.reset()`.

- Parallel-seed CHS tuning idea (from a pre-cleanup planning doc):
  the 30 seeds inside one Optuna trial (10 buildings × 3 seeds) are
  currently evaluated sequentially in `baselines/tune_ppo.py::_run_sweep`,
  ~42 h/trial. Fanning them out across 48 CPUs with
  `ProcessPoolExecutor` collapses a trial to ~1.5 h on a single node,
  eliminating the three failure modes of the 75-worker SLURM array
  (node packing, Optuna `ReservationRaceCondition` on concurrent
  `suggest()`, slow-node timeout). EnergyPlus is process-safe in
  this codebase (`train_cross_domain.py` already uses `mp.spawn`).
  Not implemented; consider when the next round of PPO tuning starts.

---

## Status snapshot (2026-05-25)

Branch: `feature/reward-normalization`, far ahead of
`origin/dev`, not yet pushed. The 2026-05-25 brainstorm
reorganized the remaining work as
**M → D-house → T → F → R → C** (see `TODO.md` § "Execution
order"); the L-section's old 4-PR-stack plan is superseded.

Research deliverables:

| Area | Status |
| --- | --- |
| New reward formula + types + 9-task preset grid | ✓ Landed |
| Calibration YAML (`reward_normalizers.yaml`) | ✓ Locked (random-policy reference) |
| RL obs/action normalization wiring (`make_rl_env_fn`) | ✓ Landed; PPO/SAC/dyn-adapt all go through it |
| EnergyPlus resource leak (B0/B0.1) | ✓ Fixed upstream (`vtaboga/minergym@956c3e1`). Residual ~14 MB/cycle native RSS growth irreducible — see § Operational gotchas |
| SAC critic-stability fixes (B1) | ✓ Landed (`64a4eb5`). B2/B3/B4 cancelled — final agents use B1 SAC defaults + committed `ppo.yaml` |
| Legacy reward family (`task1`–`task5`, `BarrierReward`, un-norm `DeadbandReward`) | ✓ Deleted (D2, `574baaa`). Presets renamed to `task_<mode>_<e0/emed/ehigh>` |
| OfficeMedium OA-mixer in action space | **Pending (Phase M)** — must land before C1; reference fix in `../RL2GNNs`; invalidates HF dataset, tuned RBCs, OfficeMedium normalizers |
| Empirical `emed`/`ehigh` study | **Pending (Phase R)** — appendix-quality result; SAC reward-distribution sweep on `test_small` subset |
| `baseline_returns.csv` | Old reward + old action space; regen on full split as **C1** under locked R values |
| PPO + SAC specialists | Old reward; rerun as **C2** on `test_small`, 1 seed, default params, 9-task grid |
| Dynamics adaptation + cross-domain (paper §6.1 / §6.2) | Re-run as final batch of Phase C (**C3 + C4**, confirmed in scope 2026-05-25) |

OSS-readiness deliverables:

| Area | Status |
| --- | --- |
| `REPRODUCING.md` | ✓ Drafted (D5, `45799f3`); kept in lock-step with code per Cross-phase principle 2; D16 is the final parity sweep |
| Eval bug fixes (`eval_ppo.py`, `eval_dynamics_adaptation.py`) | ✓ Fixed (D3, `de258ae`) |
| `baselines/requirements.txt` removed | ✓ Done (D4, `f6ec020`); install path is `pip install -e ".[training]"` |
| Public API surface defined (`design_doc.md` §3.2) | Done; **needs `CHANGELOG.md` + deprecation policy** (D13) |
| API-contract tests (Tier 1) | Exist but unlabelled; **needs `@pytest.mark.api_contract`** (D12) |
| Test-suite triage (precedes CI) | Pending (D6.5); current `pytest -m quick` has pre-existing failures |
| LICENSE | Missing (D1) |
| CI | None (D7) |
| Tutorials / docs migration to normalized preset names | Pending (D9) |
| Docs site (benchmark-problem pages, etc.) | Pending (D14) |
| `analysis/` → `baselines/` migration | Pending (audit in F2, execution in D15) |

---

## Repository hygiene snapshot (2026-05-12)

Audit done before purging the working tree. None of the items below
are in git history; this section records what was on disk so a future
reader can find the trail if needed.

### Deleted in the purge

- `scrap/` (11 GB): PPO/RBC reruns, AshraeAirLoop test, dataset-gen
  outputs, eval_g36, baseline_rollouts. No tracked code referenced it.
- `staging/` (1.5 GB): the 6 building-archetype `*.zip` files +
  `metadata.parquet` + `splits.json` used to upload the dataset to
  HuggingFace. Re-downloadable from HF; nothing in code references
  this directory.
- `analysis/{office,restaurant,singlefamily}*_tuned*/` (805 MB across
  6 dirs): per-building tuned-controller `.npz` trajectories +
  EnergyPlus outputs. Regenerable via `baselines/tune_controller.py`
  + `run_reactive_control.py`. Top-level `report.json` files retained
  where they capture audit conclusions in `analysis/TUNING_AUDIT_REPORT.md`
  (which is also being archived into this section — see below).
- `analysis/_sfh_0014_sweep/`, `analysis/task_study/{reward_design,
  reward_design_normalized,reward_design_normalized_sac,
  reward_distributions,sac_diagnostic/plots,setpoint_schedules}/`
  (~22 MB): generated plots; regenerate per Phase A3 / Phase B2.
- `analysis/SUMMARY_WORST_PERFORMERS.md`, `analysis/TUNING_AUDIT_REPORT.md`,
  `analysis/_tuning_audit.json`: outputs of the controller-tuning
  audit. Key conclusion to retain: the 5 worst RBC-tuned buildings
  are all `RestaurantFastFood-3*` in CZ 1; verdict was *HVAC-limited*
  (max actuator saturation ≥ 50%, mean |T−setpoint| during saturation
  > 1 °C), not policy-suboptimal. Read: the RBC tuning is fine; those
  buildings have undersized HVAC.
- `summaries/` (4 .md files): distilled into this `notes.md`; safe
  to delete per TODO D11.
- `plans/parallel_seeds_single_node.md`: planning note for parallel
  SAC seed runs; superseded by `baselines/scripts/train_sac_array.sh`.
- `refactoring/` (4 files): historical refactoring notes; the two
  refactoring PRs already landed on `dev` (commits 1f57775,
  6ba99ea, fdd04e0).
- `.cursor/plans/*.plan.md` (3 files): stale per-task planning docs
  superseded by `TODO.md`.
- `site/` (mkdocs build output; regenerable from `mkdocs build`).
- `logs/`, `wandb/run-*/`, all `__pycache__/`, `*.egg-info/`,
  `.pytest_cache/`, `.venv/`: regenerable caches and run artifacts.

### Promoted out of "ignored" status

- `baselines/configs/wandb/default.yaml`: required by Hydra
  (`defaults: - wandb: default` in `baselines/configs/config.yaml`);
  was caught by the `wandb/` ignore rule. `.gitignore` rule
  tightened to `/wandb/` to only match the top-level run-output dir.
- `baselines/analysis/analyze_office{medium,small}_control.py` +
  `__init__.py`: caught by the `analysis/` ignore rule, but these are
  per-building tuned-controller analysis scripts that live under
  `baselines/`. `.gitignore` rule tightened to `/analysis/` (top-level).

### Pending decision (separate task, not in this purge)

- `analysis/task_study/` (~8 KLOC of Python) is currently ignored
  yet load-bearing: tests import from it, the calibration loader
  references it by module path, and the committed YAMLs cite it as
  their source. User direction: move under `baselines/calibration/`
  (cleaned and purged), avoiding code duplication. The duplication
  to watch is the `run_training` / `run_training_sac` functions in
  `reward_design_comparison{,_sac}.py`, which fork from
  `baselines/train_{ppo,sac}.py` to add per-step reward-decomposition
  logging (`temp_penalty/τ_T`, `power_penalty/τ_E`). Refactor target:
  extract that decomposition into a callback in
  `baselines/utils/callbacks.py` so a single training entrypoint can
  cover both production and calibration use cases.
- `scripts/` (top-level): contains `diagnostics/audit_tuning_studies.py`,
  `processing/{build_single_family_dataset,merge_datasets,
  patch_sfh_schedules,process_single_family_houses,stage_from_hf}.py`,
  and 2 SLURM launchers. Currently ignored by `/scripts/`. Pending
  decision: are these maintained as part of the OSS release, or were
  they one-offs to build the dataset on HF? If maintained, untrack
  `/scripts/` and add selectively.
