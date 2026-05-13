<!-- -*- mode: markdown -*- -->

# Notes

Compact, in-flight working notes. Read this **before** starting any task.
Update as the project moves; keep it short and context-efficient.

Companion: `design_doc.md` (slow-changing intent + roadmap),
`TODO.md` (atomic action items), `paper/main.tex` (final results).

Distilled from `summaries/` (which will be deleted once everyone agrees
this file captures what is still relevant).

---

## Active decisions pending

### Task preset names

Current names are mechanical (`task_const_w0`, `task_occ_wmed`, ...).
Once legacy `task1`–`task5` is deleted, free hand to rename. Open
question: `comfort_only` / `balanced` / `energy_priority` ×
`const` / `occ` / `rand`? Defer until D9 (documentation pass).

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
mode ∈ {`const`, `occ`, `rand`}; level ∈ {`w0`, `wmed`, `whigh`}
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
- `task_occ_w0` has `w_E = 0`, so the dominance-ratio plot is NaN
  by construction — expected, not a bug.
- **EnergyPlus resource leak (B0).** Long-running processes leak both
  `eplus_output_dir` contents (fills `$SLURM_TMPDIR` / `$SCRATCH`,
  eventually triggers no-space-left or inode/file-count errors) and
  EnergyPlus simulation threads + native state (RAM grows linearly
  with envs created). Currently masked by
  `baselines/utils/evaluation.py::close_env_aggressively`; any new
  entrypoint that forgets to call it leaks silently. Root cause is in
  `minergym` (`EnergyPlusEnvironment.close` is the inherited
  `gym.Env` no-op). See `TODO.md` § B0.
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

## Status snapshot

Branch: `feature/reward-normalization`, 5 commits ahead of
`origin/dev`, not yet pushed.

Research deliverables:

| Area | Status |
| --- | --- |
| New reward formula and types | Landed |
| Calibration YAML | `building2building/data/reward_normalizers.yaml` (random-policy) |
| 9-task preset grid | Landed |
| RL obs/action normalization wiring | Landed; PPO/SAC/dyn-adapt all use `make_rl_env_fn` |
| PPO under new reward | Trained on winter only; freezes at `w_E ≥ 10`, `target_kl=0.02` too tight; **full-year sweep + retune pending** (B3, B4) |
| SAC under new reward | `sac.yaml` updated with critic-stability fixes. **Conflicts with under-exploration narrative in § SAC; reconcile before Phase B closes** (B1, B2) |
| EnergyPlus resource leak | **Partially patched** by `close_env_aggressively`. Root cause in `minergym`. **B0: fix upstream + leak-free default `close()`** before B2/B3/B4 and Phase C |
| Legacy reward family (`task1`–`task5`, `BarrierReward`, un-norm `DeadbandReward`) | Still present; **deletion before paper rerun** (D2) |
| Paper figures and tables | Old reward; rerun pending (Phase C) |
| `baseline_returns.csv` | Old reward; regen post-deletion (C1) |

OSS-readiness deliverables:

| Area | Status |
| --- | --- |
| Public API surface defined (`design_doc.md` §3.2) | Done; **needs `CHANGELOG.md` stub + deprecation policy** (D13) |
| API-contract tests (Tier 1) | Exist but unlabelled; **needs `@pytest.mark.api_contract`** (D12) |
| Pure-logic tests (Tier 2) | Exist; well-covered |
| Integration tests (Tier 3) | Exist; under-documented |
| LICENSE | Missing (D1) |
| Eval bug fixes (`eval_ppo.py`, `eval_dynamics_adaptation.py`) | Partially edited; regression tests pending (D3) |
| CI | None (D7) |
| Tutorials | 5 `.py` + 5 `.md`; **drifted; uses legacy `task1` family** (D9) |
| `REPRODUCING.md` | Pending (D5) |
| Docs site | Builds; uses legacy task names; needs reward + benchmark-problem pages (D9, D14) |

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
