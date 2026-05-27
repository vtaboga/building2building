<!-- -*- mode: markdown -*- -->

# Notes

Compact, in-flight working notes. Read this **before** starting any task.
Update as the project moves; keep it short and context-efficient.

Companion: `design_doc.md` (slow-changing intent + roadmap),
`TODO.md` (atomic action items), `paper/main.tex` (final results).

Distilled from the now-deleted `summaries/` directory (content absorbed here).
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

## OfficeMedium OA-mixer fix

Design note for Phase **M** in `TODO.md` (OfficeMedium OA-mixer
action-space fix). Pins the design choices before any code change.
Source of truth for items **M1–M4**. The five questions in `TODO.md`
§ M0 are answered below; the supporting exploration of the B2B
pipeline and the RL2GNNs reference is summarized inline.

### Problem in one line

`make_vav_system_controllable` in
`building2building/pipeline/actuators.py` does not emit any
outdoor-air-mixer actuator for OfficeMedium. Current agent action
dim for OfficeMedium is **33** (3 SAT + 15 flow + 15 htg; 15 clg
fixed at 40 °C via `hvac_action_space`). RL2GNNs's reference fix
exposes **3 OA mass-flow** actuators (one per air loop) — for B2B
we add the same 3 actuators, bringing OfficeMedium to **36** agent
actions.

### Q1 — Per-air-loop multiplicity across the 1000 OfficeMedium

**Answer: enumerate OA mixers dynamically from the epJSON
ontology, mirroring how the existing VAV actuators are emitted.
Nothing about loop count is hardcoded.**

- The existing VAV actuator emission in
  `make_vav_system_controllable`
  (```877:1165:building2building/pipeline/actuators.py```) is
  fully topology-driven: it discovers `AirLoopHVAC` instances
  via an RDF/SPARQL walk (`?loop a "AirLoopHVAC"`, line 949) and
  installs SAT + per-zone actuators for whatever it finds. The
  new OA mixer emission follows the **same pattern**: one
  additional SPARQL query that walks
  `AirLoopHVAC` → `AirLoopHVAC:OutdoorAirSystem` →
  `Controller:OutdoorAir`, and emits one
  `ActuatorDescription` per controller found.
- The DOE Reference OfficeMedium prototype is structurally a
  3-floor / 3-VAV building (`VAV_1`/`VAV_2`/`VAV_3` in the
  RL2GNNs epJSON, ```2:28:../RL2GNNs/officerl/data/building.epjson```
  with matching `Controller:OutdoorAir` objects at
  ```5389:5435:../RL2GNNs/officerl/data/building.epjson```), and
  B2B's 1000 OfficeMedium rows are parametric variants of the
  same DOE prototype (sourced from
  `vtaboga/multizones_reference_buildings`,
  ```131:138:building2building/sources/multizones_reference_buildings.py```).
  In practice, all 1000 will yield 3 OA actuators. **But the
  pipeline does not encode this expectation.** If a variant
  ever has 2 or 4 loops the code emits 2 or 4 OA actuators.
- `metadata.parquet` already carries a per-row `action_dim`
  field (```147:147:building2building/data/registry.py```), so
  downstream consumers see whatever count the pipeline emitted
  and do not need to know "OfficeMedium = 3 OA actuators".

### Q2 — Actuator scheme, action range, units, and the `Availability Status` question

**Answer: expose 3 × `(Outdoor Air Controller, Air Mass Flow Rate)`
to the agent; do NOT expose `Availability Status` (diverge from
RL2GNNs).**

- Units: **kg/s** mass flow, matching the EnergyPlus
  `Outdoor Air Controller × Air Mass Flow Rate` actuator and the
  RL2GNNs metadata (`[kg/s]`, ```421:444:../RL2GNNs/officerl/data/metadata.json```).
- Action range: **`[0.0, 5.0]` kg/s** per loop. Rationale: the
  RL2GNNs metadata uses `[0.0, 10.0]` but the tuned RBC
  saturates well below that (`oa_flow_max: 8.5`,
  `oa_flow_min/oa_neutral: 1.37` kg/s in
  ```41:47:../RL2GNNs/configs/policy/baseline.yaml```). The DOE
  OfficeMedium prototype's autosized minimum outdoor-air flow
  rate is approximately 1.1–1.4 kg/s; the design supply-air flow
  per loop is ~4–6 kg/s. **5.0 kg/s gives headroom > design
  flow** (so the actuator can dominate the mixer) without
  inflating the agent's exploration range to physically
  unattainable values. The bound is conservative; widen later
  only if a controlled experiment shows agents saturating at
  5.0.
- `AirLoopHVAC × Availability Status`: **not exposed.** RL2GNNs
  exposes it (3 extra agent actions in
  ```235:238:../RL2GNNs/officerl/action_spaces.py```), but the
  B2B regime keeps every actuator's enable-side **pinned on**
  so the agent's writes are never silently overridden by a
  schedule or `AvailabilityManager`. The current code already
  enforces this for the supply fan and the `AirLoopHVAC`
  itself via `_ensure_always_on_availability` and
  `_set_fan_always_on`
  (```1082:1144:building2building/pipeline/actuators.py```).
  **M1 extends this regime to the OA-mixer side** so the new
  OA actuator is similarly authoritative:
  1. The `Controller:OutdoorAir` object's `schedule_name`
     (and `minimum_outdoor_air_schedule_name`) must not
     reference any non-trivial schedule that EnergyPlus would
     apply on top of the EMS actuator write. M1 must rebind
     these to an always-on constant schedule (or strip them)
     and assert post-write that the controller has no
     overriding schedule field set.
  2. Any `AvailabilityManager:Scheduled` /
     `AvailabilityManager:NightCycle` attached to the
     OA-system loop must continue to point to the always-on
     schedule installed by the existing VAV path (already
     handled by `_ensure_always_on_availability`).
  3. M1's regression test
     (`tests/quick/test_officemedium_actuator_set.py`)
     asserts that on the minimal VAV fixture, after
     `make_controllable`, **no schedule, EMS program, or
     availability manager exists that can override any of
     the agent-facing actuators** — SAT, flow, htg, clg, **or
     the new OA mass flow**. This is the "every actuator
     always on" acceptance criterion.
- **Documented divergence from RL2GNNs:** B2B's action space
  has no `Availability Status` entry. The pin-everything-on
  invariant is asserted in test rather than left to the agent
  to learn.
- Resulting OfficeMedium agent action layout (per `VAVSystem`,
  outer-product over loops):
  - 1 × SAT (Schedule:Constant, `[C]`, `[10, 55]`)
  - n_zones × flow fraction (`[frac]`, `[0, 1]`)
  - n_zones × heating setpoint (`[C]`, `[10, 35]`)
  - n_zones × cooling setpoint (`[C]`, `[18, 40]`) — fixed at 40 °C
  - **1 × OA mass flow (`Outdoor Air Controller`,
    `Air Mass Flow Rate`, `[kg/s]`, `[0, 5]`)** ← new in M1
- Total for 3 loops × 5 zones: `3 + 15 + 15 + 15 + 3 = 51`, of
  which **15 cooling setpoints are fixed**, giving
  **agent_action_dim = 36** (up from 33).

### Q3 — Reactive-controller pin value

**Answer: pin the OA mixer at a constant `1.37 kg/s` per loop in
the B2B reactive controller; do NOT search this knob with Optuna.**

- The RL2GNNs RBC is **not a single immutable constant** — it
  runs an economizer when the loop overheats and outdoor air is
  cool, otherwise it falls back to `oa_flow_min` (see
  ```298:323:../RL2GNNs/alg/hvac_baseline/baseline_controller.py```).
- The fallback value `oa_flow_min = oa_neutral = 1.37 kg/s` in
  the tuned RL2GNNs config (```41:47:../RL2GNNs/configs/policy/baseline.yaml```)
  was chosen to match the DOE OfficeMedium autosized minimum
  ventilation rate (~1.12 m³/s × 1.225 kg/m³ ≈ 1.37 kg/s).
- **Decision for B2B:** the reactive controller writes
  **`1.37 kg/s` unconditionally** (no economizer). Rationale:
  - The TODO M0 brief explicitly says "hold the OA mixer at a
    constant value" and "OA mixer command is held at the constant
    value pinned by M0 (not Optuna-searched)" (TODO § M3).
  - The agent's job — once the OA mixer is in the action space —
    is to learn the economizer behavior. The RBC stays a
    deliberately simple "always at minimum OA" floor, against
    which agent learning is measured.
- **Codification:** the constant becomes a new field
  `oa_mass_flow: float = 1.37` in the `AirLoopConfig` dataclass
  in `baselines/controllers/air_loop.py`. It is **not** declared
  in the Optuna search space in `baselines/tune_controller.py`
  (`_suggest_air_loop`, ```71:97:baselines/tune_controller.py```).
  The new YAMLs written by M3 carry the constant as a fixed
  field.

### Q4 — Pipeline change locus

**Answer: change only `make_vav_system_controllable` in
`building2building/pipeline/actuators.py`. No change to
`extract_discovery_metadata`.**

- `make_vav_system_controllable`
  (```877:1165:building2building/pipeline/actuators.py```) is the
  only function that emits the OfficeMedium actuator list. The
  fix is local: extend the per-loop installation block
  (```1134:1164:building2building/pipeline/actuators.py```) with
  an `install_oa_mixer_actuator(loop_name)` helper that:
  1. Looks up the loop's `AirLoopHVAC:OutdoorAirSystem` via the
     existing RDF ontology pattern.
  2. Resolves the `Controller:OutdoorAir` object it references.
  3. Returns an `ActuatorDescription(component_type="Outdoor Air
     Controller", control_type="Air Mass Flow Rate",
     component_name=<controller name>, units="[kg/s]",
     lower_bound=0.0, upper_bound=5.0)`.
- The `VAVSystem` dataclass (```778:790:building2building/pipeline/actuators.py```)
  gains one field — `oa_mass_flow: ActuatorDescription` — and
  appends it in `actuator_descriptions()`. This is a schema
  change to `equipment.json`; old `equipment.json` files
  serialized without the new field will fail `cattrs.structure`
  on load (per Q5: this is the desired fail-loud behavior, no
  compat shim).
- `extract_discovery_metadata`
  (```117:197:building2building/pipeline/discovery.py```) extracts
  only `(source_path, net_conditioned_area, warmup_phases,
  warmup_days)` (see the `Metadata` dataclass,
  ```24:32:building2building/pipeline/discovery.py```); none of
  these depend on the actuator inventory. **No code change
  required.** The one indirect pre-condition is that the new OA
  actuator registration must not crash the 1-day discovery
  simulation that runs immediately after `make_controllable`
  (per ```58:115:building2building/pipeline/__init__.py```).
  Registering the standard E+
  `Outdoor Air Controller × Air Mass Flow Rate` actuator is
  expected to be benign, but **M1's dry-run must invoke the full
  `create_complete_pipeline`** (not just `make_controllable` in
  isolation) on one real OfficeMedium epJSON to confirm
  discovery still passes end-to-end.
- The agent-action-dim plumbing in `simulator/__init__.py`
  (```195:296:building2building/simulator/__init__.py```) is
  topology-agnostic — it counts whatever `actuator_descriptions()`
  returns. The new OA actuator flows through automatically; the
  per-row `action_dim` field in `metadata.parquet` (built in M2)
  will reflect the new count.
- Pipeline dry-run on one real OfficeMedium epJSON pulled from
  the HF dataset is required before M1 commits — verify that
  the actuator list contains exactly one new
  `Outdoor Air Controller × Air Mass Flow Rate` entry per air
  loop, that everything else is bit-identical, and that the
  full pipeline (`create_complete_pipeline`) completes through
  the discovery sim.

### Q5 — Backward compatibility with the existing HF dataset

**Answer: no backward compatibility. Push the regenerated
OfficeMedium slice to `main` on `vtaboga/building2building_dataset`;
the existing `REVISION = "main"` pin in `download.py` automatically
picks it up. Old downstream artefacts (tuned RBC YAMLs, reward
normalizers, baselines) are regenerated in M3/M4 and are not
expected to load against the new dataset.**

- Current pinning: `REPO_ID = "vtaboga/building2building_dataset"`
  with `REVISION = "main"` in `building2building/data/download.py`
  (```19:20:building2building/data/download.py```). M2 pushes to
  `main`; no `download.py` change.
- **Failure mode on the old schema:** a user with a stale local
  HF cache and the new code will see a `cattrs`
  `ClassValidationError` from `structure(..., list[AnyEquipment])`
  (```1219:1223:building2building/pipeline/actuators.py```)
  complaining about the missing `oa_mass_flow` field on
  `VAVSystem`. That is the desired fail-loud behavior per
  `AGENTS.md`. The fix is "clear your HF cache" / re-pull from
  `main`. **No silent fallback** to the old action space; no
  compat shim in the dataclass.
- Old downstream artefacts touched by M3/M4 are overwritten in
  place (no `_v2` files) and are committed alongside the M-phase
  commits, so a `git checkout` at any post-M commit yields a
  consistent (dataset revision, RBC YAMLs, reward normalizers)
  tuple. Anyone running an older commit must `pip install` the
  matching older B2B version, which still resolves `REVISION =
  "main"` and now gets the new dataset — but at that older commit
  the old `equipment.json` schema is still expected, so it will
  fail to structure. **Documented limitation:** running an
  arbitrary historical commit against `main` HF is not
  supported.
- Documentation: `REPRODUCING.md`, `docs/guide/buildings.md`,
  and `CHANGELOG.md` (D13) note the action-space change as a
  one-way breaking change; no per-revision instructions are
  needed because the pin stays at `main`.

### Out-of-scope (called out so it's not silently dropped)

- **RL2GNNs cross-repo parity export.** RL2GNNs's
  `officerl/data/metadata.json` is a hand-curated flat list. The
  comment in ```96:96:../RL2GNNs/officerl/env.py``` says "Run the
  export script from Building2Building". That export script is
  not in the tracked B2B tree and is not part of the M-phase
  scope. The OA actuator is added to B2B's `equipment.json`
  format only.
- **`make_controllable(controls=...)` cleanup.** The `controls`
  parameter on `make_controllable`
  (```1192:1225:building2building/pipeline/actuators.py```) is
  dead code (declared, never read; residential code passes it).
  Deletion belongs to Phase D-house, not M1.
- **Availability Status as agent action.** Documented divergence
  from RL2GNNs (Q2); not added in this phase.
- **Economizer logic in the RBC.** RL2GNNs's RBC has it; B2B's
  RBC stays at the constant `oa_flow_min` pin (Q3). Adding
  economizer to B2B's RBC is a separate decision; if anyone
  proposes it, it must include a new Optuna search and a
  re-tune of all 8 CZ YAMLs, which is exactly what Q3 chose to
  avoid.

### Touch-list summary (forward reference for M1–M4)

- **M1 (code change):**
  - `building2building/pipeline/actuators.py`:
    - `VAVSystem` dataclass: new `oa_mass_flow:
      ActuatorDescription` field, appended in
      `actuator_descriptions()`.
    - `make_vav_system_controllable`: new
      `install_oa_mixer_actuator(loop_name)` helper that
      (a) discovers the loop's `Controller:OutdoorAir` via a
      SPARQL walk through `AirLoopHVAC:OutdoorAirSystem`,
      (b) rebinds any schedule fields on that controller to the
      always-on constant schedule already installed by the
      existing fan/availability path, and (c) returns the
      `ActuatorDescription`. Per-loop block (currently
      ```1134:1164:building2building/pipeline/actuators.py```)
      gains one call to the helper and passes the result into
      `VAVSystem(...)`.
  - `tests/quick/test_officemedium_actuator_set.py` (new
    regression test pinned against the `minimal_vav/` fixture):
    asserts (a) the actuator list contains exactly one new OA
    actuator per air loop, (b) no schedule, EMS program, or
    `AvailabilityManager` can override any agent-facing
    actuator (SAT, flow, htg, clg, OA).
- **M2 (data regen, no code unless `regen_dataset.py` does not
  yet exist):**
  - `building2building/pipeline/regen_dataset.py` (new — entry
    point per Cross-phase principle 3).
  - `building2building/pipeline/scripts/regen_officemedium.sh`
    (Slurm wrapper).
  - HF dataset push to `vtaboga/building2building_dataset@main`
    (no revision bump; `download.py` unchanged).
- **M3 (re-tune):**
  - `baselines/controllers/air_loop.py`: add
    `oa_mass_flow: float = 1.37` to `AirLoopConfig`, write it
    to the OA actuator index in `predict()`.
  - `baselines/tune_controller.py`: **no change** (OA is not
    in the search space).
  - `baselines/configs/tuned_controllers/air_loop_officemedium_cz{1..8}.yaml`:
    regenerated (8 files overwritten in place — no `_v2`).
- **M4 (recalibrate normalizers):**
  - `building2building/data/reward_normalizers.yaml`: rewrite
    **every row** (all building types, all CZs, all three
    seasons).  The original M4 intent was to recompute only the
    OfficeMedium rows, but auditing the calibration scripts in
    response to this turn revealed that they still hard-coded
    the legacy preset name `task3` (deleted in D2) and recovered
    `temp_penalty` by inverting the un-normalized deadband-reward
    formula (`temp_penalty = -reward - w_E * power_penalty`),
    which is invalid against the only surviving reward,
    `NormalizedDeadbandReward`.  Both
    `analysis/task_study/compute_random_policy_reward_normalizers.py`
    and its tuned-RBC twin `compute_reward_normalizers.py` now
    pass `task_occ_emed` (the post-D2 semantic successor to
    `task3`: same occupancy + seasonal-unoccupied schedule, same
    `dT = 1.0`; the `energy_weight` is irrelevant because the
    scalar reward is no longer inspected) and recompute the
    `(temp_penalty, power_penalty)` decomposition directly from
    `info["raw_observation"]` via
    `building2building.simulator.rewards._deadband_components`,
    using the comfort zones, target schedule and `dT` from the
    live `env.unwrapped.reward_fn` so the recomputed components
    match exactly what training/eval will see.
  - **Cache invalidation:** the per-building rollout cache under
    `$SCRATCH/b2b_reward_normalizers_random/data/<run_period>/<bt>/<bid>.json`
    is keyed by `(run_period, bt, bid)` only -- no schema marker
    -- so every cached JSON predating this commit is stale.
    `REPRODUCING.md` § B/A documents the one-time full-cache wipe
    + full 48-task `sbatch` + login-node `--mode aggregate`
    sequence.
  - **Outcome:** the `git diff` on `reward_normalizers.yaml`
    will touch every row, not just OfficeMedium's.  This is the
    correct behaviour; the previous numbers for all 6 building
    types were computed via the now-known-buggy reward-arithmetic
    recovery and should not be carried forward as cached
    technical debt.
  - Path migration of `compute_random_policy_reward_normalizers`
    → `baselines/compute_reward_normalizers` is **F2 / D15
    territory**; M4 uses whichever path is current at the time.
    If F2 has not yet moved it, M4 runs the analysis-path
    invocation and files no migration sub-commit.
  - This commit ships the code + docs update; the YAML update
    lands as the follow-up commit after the user runs the Slurm
    array and the aggregate step.

## D10 — pyright status (2026-05-26)

Running `pyright building2building baselines` (after black pass) reports
**242 errors** (all pre-existing, none introduced by D10). Fixing them all
is out-of-scope for the OSS release; they are catalogued here so a
contributor can triage later.

**Root causes (by frequency):**

1. **`matplotlib`/`torch` private-import stubs** (~80 errors): pyright
   treats `matplotlib.pyplot.Figure`, `matplotlib.pyplot.Axes`,
   `torch.cat`, `torch.tensor`, `torch.Size`, etc. as private.  Fix:
   import from the canonical locations (`from matplotlib.figure import
   Figure`; `import torch; torch.cat(...)` is fine, but type annotations
   should use `torch.Tensor` etc.).

2. **`str` coerced to `BuildingType` Literal in baselines** (~40 errors):
   Hydra deserialises config fields as plain `str`; baselines then pass
   these directly to typed API calls expecting
   `Literal['SingleFamilyHouse', ...]`.  Fix: add explicit casts or
   validate with `BuildingType(value)`.

3. **`PPO` not assignable to `PolicyLike`** (~5 errors): The
   `PolicyLike` protocol does not cover SB3 `PPO` models; needs a union
   or a shim.

4. **Pipeline / sources / store type gaps** (~120 errors): scattered
   issues in `pipeline/actuators.py`, `simulator/wrappers.py`,
   `sources/*.py`, `store.py`.  Many are consequence of `cattrs`
   returning `Unknown` types or untyped third-party objects entering
   the pipeline.  Lower priority; would require per-file triage.

**Decision 2026-05-26:** leave as-is for the OSS release.  The public
API surface (`building2building/__init__.py.__all__`) is clean; most
errors are in pipeline internals and baselines scripts.  Track remaining
issues under future Phase T / Phase F follow-ups.

---

