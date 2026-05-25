<!-- -*- mode: markdown -*- -->

# TODO

Atomic action items toward open-source release and paper rerun under the
new normalized reward. **One TODO, one commit** (per `AGENTS.md`).

Items are tagged by phase (A/B/C/D, see `design_doc.md` § Roadmap) but
otherwise unordered — the user reorders / picks priority. The only hard
cross-phase dependency is **D2 must precede C1**.

Format per item: short title; affected files; acceptance check;
references.

---

## Phase A — Finalize the reward

### ~~A1. Decide the calibration controller and document it~~ ✓

Choice: `reward_normalizers.yaml` (SAC-warmup uniform random; formerly
`reward_normalizers_random_linear.yaml`). Rationale: policy-independent;
bakes in no RBC-specific bias. Tuned-RBC YAML deleted.

### ~~A2. Re-run calibration if A1 invalidates the current YAML~~ N/A

A1's choice matches the pre-existing default; no re-run needed.

### ~~A3. Validate and commit the calibration sanity plot~~ ✓

Plot lives in the git-ignored `/analysis/` tree; location documented in
`notes.md` § "Calibration sanity plot". Median/τ = 1.0 by construction;
IQRs match committed YAML. Regenerate with
`python -m analysis.task_study.compute_random_policy_reward_normalizers --mode aggregate`.

---

## Phase B — Stabilize RL training under the new reward

### ~~B0. Diagnose and fix the EnergyPlus resource leak~~ ✓

**Done.** Commits `be7463c` (initial fix), `5118af5` / `bd6a509` /
`aab1843` / `7dd389f` (B0.1 follow-ups). All acceptance criteria met:

- `B2BEnergyPlusEnvironment.close()` is leak-free: joins the EnergyPlus
  thread, calls `gc.collect()` so `ManagedState`'s `weakref.finalize`
  fires, and `rmtree`s the tracked output directory — all gated on
  `had_simulation = self.ep is not None` so repeated no-op calls are
  safe.
- `B2BEnergyPlusEnvironment.reset()` is leak-free: calls `self.close()`
  (thread join + rmtree), recreates the output dir with `mkdir`, then
  delegates to `super().reset()` (which calls `self.close()` again
  polymorphically; the `had_simulation` guard prevents the second call
  from deleting the freshly recreated dir).
- `close_env_aggressively` downgraded to a `DeprecationWarning` shim;
  no callers remain outside the shim definition.
- `psutil` added to `[test]` extras; RSS guard test no longer skips.
- Upstream fixes at `vtaboga/minergym@6d03b9a` pinned in
  `pyproject.toml`.
- Tests: `tests/long/test_env_leak.py` (`TestEnvLeakClose` +
  `TestEnvLeakReset`); gated on `long` marker (require EnergyPlus).

Residual known item: ~14 MB/cycle irreducible EnergyPlus-native RSS
growth (C++ globals in the DLL). Not a Python-side leak; subprocess
isolation is the only complete fix (deferred to a future phase).
See `notes.md` § "Operational gotchas" for details.

### ~~B0.1. Close the residual leaks from B0~~ ✓

**Done.** All sub-tasks completed; see commit log (`5118af5`,
`bd6a509`, `aab1843`, `1832dab`, `236c3df`, `7dd389f`).

- ~~B0.1.a~~ ✓ `reset()` override added; `had_simulation` guard in
  `close()` prevents the double-delete from upstream's polymorphic
  `self.close()` call inside `reset()`. `TestEnvLeakReset` added.
- ~~B0.1.b~~ ✓ Test docstring fixed (`<50 MB` → `_RSS_MAX_GROWTH_BYTES`).
- ~~B0.1.c~~ ✓ 14 MB/cycle figure documented in `notes.md`;
  `_RSS_PER_CYCLE_BYTES = 16 MB` (14 native + 2 margin) is the bound.
  A formal SLURM measurement to derive `p95` empirically is deferred
  (low priority; bound is already conservative).
- ~~B0.1.d~~ ✓ Already done in `be7463c`; `tune_controller.py` uses
  `b2b_tune_{bid}_` prefix throughout.
- ~~B0.1.e~~ ✓ No `close_env_aggressively` callers outside shim definition.
- ~~B0.1.f~~ ✓ Deviation recorded in `notes.md`.
- ~~B0.1.g~~ ✓ Constructor kwarg present; docstring already correct.
- ~~B0.1.h~~ ✓ `try/except` removed; `try_stop()` called bare.
- ~~B0.1.i~~ ✓ `psutil` added to `[test]` extras.

#### ~~B0.1.upstream — Cleaner fixes to land on the `minergym` fork~~ ✓

**Done.** Upstream commits `6d03b9a` and `956c3e1` on
`vtaboga/minergym`; pinned in `pyproject.toml`. The in-tree
`B2BEnergyPlusEnvironment` subclass was removed entirely;
`create_simulator()` constructs `EnergyPlusEnvironment` directly with
`cleanup_output_dir_on_close=eplus_output_dir is not None`. The
historical spec below is retained for the audit trail.

The following are *cleaner* if done upstream than worked around
in-tree. They are listed in priority order. Each is small and
self-contained; bundle them as one PR if possible. The in-tree
counterparts (B0 + B0.1.a + B0.1.g) should be downgraded to thin
shims once the upstream changes ship and a new minergym version is
pinned in `pyproject.toml`.

**Constraints (re-stated for the fork worker):**
1. Public Gymnasium interface (`reset`, `step`, `close`) and the
   `EnergyPlusEnvironment` constructor signature must not change.
2. Per-step overhead must remain at zero (no new work in
   `EnergyPlusSimulation.step`).
3. Per-reset overhead must stay below ~1 s (a `Thread.join(timeout)`
   and a `gc.collect()` are fine; no subprocess work).
4. No silent `try`/`except` swallowing — match this repo's
   "fail loudly" policy.

**1. `EnergyPlusEnvironment.close()` override.**
- File: `minergym/environment.py`.
- Replace the inherited `gym.Env.close()` no-op with:
  - `if self.ep is not None`: capture `ep_thread` from
    `self.ep.state` *before* `try_stop()` (see in-tree comment for
    why — `try_stop` transitions state to `StateDone` which drops
    `ep_thread`), call `self.ep.try_stop()`, then `ep_thread.join(
    timeout=10.0)` with a `logger.warning` if still alive, then
    `self.ep = None`. Finish with `gc.collect()` so the
    `weakref.finalize` callback for `ManagedState` runs and
    `delete_state()` is called.
- This makes the in-tree `B2BEnergyPlusEnvironment.close()` a thin
  shim that only handles `_b2b_eplus_output_dir` cleanup
  (which can't move upstream because the directory lives in
  `MakeEnergyPlus`, not on the env — see #3 below for an upstream
  fix for that).

**2. `EnergyPlusEnvironment.reset()` cleanup of the previous episode.**
- File: `minergym/environment.py`.
- Current upstream `reset()` calls `self.ep.try_stop()` but leaves
  the thread unjoined and `self.ep` referenced through
  reassignment. Refactor to call `self.close()` (the new method
  from #1) before `self.ep = self.make_energyplus()`.
- This automatically gives every `gym.Env` consumer a leak-free
  reset without any wrapper-level workaround. Removes the need for
  B0.1.a's in-tree `reset()` override.

**3. Track `eplus_output_dir` on `EnergyPlusEnvironment`.**
- Files: `minergym/environment.py`, `minergym/runtime.py`
  (`MakeEnergyPlus`).
- Currently `eplus_output_dir` is a constructor argument to
  `MakeEnergyPlus` and is not stored on the resulting
  `EnergyPlusEnvironment`. Expose it as a public attribute
  (`self.eplus_output_dir: Path | None`) populated by `__init__`,
  and have the new `close()` from #1 optionally `shutil.rmtree(
  self.eplus_output_dir, ignore_errors=True)` when set.
  - Behaviour gate: `cleanup_output_dir_on_close: bool = False`
    constructor flag, defaulting to `False` to preserve current
    upstream behaviour for users who want the artefacts.
    `building2building` would set it to `True`.
- Once shipped, the in-tree `_b2b_eplus_output_dir` plumbing
  collapses to passing `cleanup_output_dir_on_close=True` plus the
  output dir to `EnergyPlusEnvironment.__init__`.

**4. Optional: a `thread_join_timeout` constructor param.**
- File: `minergym/environment.py`.
- Make the join timeout from #1 a named parameter (default 10 s)
  so heavy users (large buildings, high `warmup_phases`) can
  raise it without subclassing.

**Acceptance for the fork worker:**
- Upstream PR opened, linked from `notes.md` § "Operational
  gotchas" and from `B2BEnergyPlusEnvironment`'s docstring.
- A version pin in `pyproject.toml` (`minergym @ git+...@<sha>` or
  a tagged release) once the PR is merged or the fork is
  consumable.
- In-tree `B2BEnergyPlusEnvironment` collapses to (at most) the
  three lines that pass `eplus_output_dir` and
  `cleanup_output_dir_on_close=True` into the upstream constructor.
- All `tests/long/test_env_leak.py` tests pass with the new
  upstream `EnergyPlusEnvironment` directly (i.e. without the
  in-tree subclass), proving the fix is complete upstream.

### ~~B1. Apply SAC fixes 1–3 to the policy config~~ ✓ done

- Files: `baselines/configs/policy/sac.yaml` —
  `log_std_init: 0.0`, `ent_coef: 0.2`, `use_sde: false`. Add an
  inline comment block citing the diagnostic findings.
- Acceptance: `pytest -m quick tests/quick/test_train_sac_smoke.py`
  passes; SAC training launcher still produces valid trajectories
  on a 50 k-step smoke test.
- Reference: `notes.md` § "SAC".
- **Commit**: `64a4eb5` (feature/reward-normalization)

### B2. SAC ablation on the calibration anchor

Run the 5-cell × 3-seed × 6-building ablation matrix on
`task_occ_wmed`, full-year, to confirm the fixes restore stable
learning. Commit a result note.

- Files: `analysis/task_study/sac_diagnostic/` (data + plots),
  update `notes.md` with the chosen final SAC config.
- Acceptance: 0/18 (or near-zero) runs show
  `best - final > 0.1`; action saturation stays below 10% at every
  checkpoint; plot in `notes.md` quick-reference table.
- **Script committed** — submit with:
  `sbatch --array=0-89 analysis/task_study/sac_diagnostic/submit_ablation.sh`
- **Status**: waiting for Slurm run; fill result table in `notes.md` § "SAC B2 ablation".

### B3. Re-tune PPO `target_kl` for the normalized regime

Test `target_kl ∈ {None, 0.05, 0.1}` on `task_occ_wmed`, full-year,
6 buildings, 3 seeds.

- Files: `baselines/configs/policy/ppo.yaml`, result note appended
  to `notes.md`.
- Acceptance: 0/6 buildings freeze in the chosen setting at `w_E = 5`;
  documented winner in `notes.md`.

### B4. Re-run PPO reward-design distribution study on `full_year`

The existing study is winter-only. The calibration regime is
full-year; without this rerun the "balance at 1.0" claim is not
directly validated.

- Files: launcher under `analysis/task_study/scripts/`, plots under
  `analysis/task_study/reward_design_normalized/plots/`. Delete
  pre-existing winter `.npz` files from `$SCRATCH/...` first
  (`notes.md` Gotcha).
- Acceptance: `fig_balance_vs_steps_full_year.png` shows red and blue
  curves near 1.0 at the calibration anchor (median building);
  `fig_dominance_ratio_vs_steps_full_year.png` is near 1.0 for
  `task_occ_wmed`.

---

## Phase C — Re-run paper experiments

**Strict prerequisite**: D2 (legacy reward deletion) lands first so
the new artifacts are computed against a single reward family.

### C1. Regenerate `baseline_returns.csv` under the new reward

Run `baselines/run_reactive_control.py` across all
`(building_type, task, run_period, building_id)` combinations under
the new normalized reward.

- Files: `building2building/scores/baseline_returns.csv` (replaced
  in full); update its header schema if needed.
- Acceptance: row count covers all train and test buildings × task
  presets × `{full_year, winter, summer}`;
  `pytest -m quick tests/quick/test_scoring.py` passes.

### C2. Re-run PPO specialists (paper §5)

- Files: `baselines/train_ppo.py` Hydra outputs collected under
  `results/`; updated CSV + figure (`fig_ppo_specialist`).
- Acceptance: figure regenerates from a single command; numerical
  values for the paper table can be cited from the CSV.

### C3. Re-run dynamics adaptation (paper §6.1)

- Files: `baselines/train_dynamics_adaptation.py`, updated
  `fig_transfer_rew` and `fig_transfer_temp_deviation`.
- Acceptance: figures regenerate cleanly; the three approaches
  (specialist, baseline, parameterized) are evaluated on the new
  reward with seeds documented.

### C4. Re-run cross-domain Amorpheus (paper §6.2)

- Files: `baselines/train_cross_domain.py`, updated
  cross-domain figure.
- Acceptance: figure regenerates cleanly under the new reward.

### C5. Update paper tables and figures

Per `AGENTS.md`, the paper is camera-ready; non-trivial changes need
explicit user approval. This TODO covers the in-scope updates:

- Files: `paper/main.tex` — Task table (replace `Task 1`–`Task 4`
  rows with the chosen normalized presets and weight levels);
  swap experiment figures; update numerical results in §5/§6 prose.
- Acceptance: PDF compiles; reviewer-flagged confound about
  cross-building `w_E` comparability is explicitly addressed in the
  prose; user has reviewed and approved the diff.

---

## Phase D — Open-source readiness

Phase D splits into two parallel tracks per `design_doc.md` §6. The
existing D1–D11 numbering is preserved (so commit messages and
references stay valid); the table below maps each item to its track.

| Track | Items | Theme |
| --- | --- | --- |
| **D-API** (researcher consumable) | D5, D8 (partial), D9, plus the new D12–D14 below | Public API surface, tutorials, REPRODUCING.md, contract tests |
| **D-house** (release housekeeping) | D1, D2, D3, D4, D6, D7, D8 (partial), D10, D11 | LICENSE, CI, legacy deletion, bug fixes, formatter passes |

New items added when Phase D was split into two tracks (2026-05-13):

### D12. Define and label the API contract test tier (D-API)

Per `design_doc.md` §4 (Tier 1): `tests/quick/test_api*.py`,
`test_new_api.py`, `test_gym_registration.py`, `test_climate_zones.py`,
`test_data_registry.py`, `test_selection_and_env_creation.py`,
`test_types.py` collectively pin the public surface listed in §3.2.

- Files: add `@pytest.mark.api_contract` marker (register in
  `pyproject.toml [tool.pytest.ini_options].markers`); add a 3-line
  comment header to each file stating "this file pins the public
  API contract; changes here = breaking API changes; requires a
  `CHANGELOG.md` entry"; expand `conftest.py` `pytest_collection_modifyitems`
  to auto-apply the marker by file glob if convenient.
- Acceptance: `pytest -m api_contract` selects exactly the files
  listed above; `pytest -m "quick and not api_contract"` selects the
  Tier-2 pure-logic tests; a deliberate breaking change to
  `building2building/__init__.py` exports causes at least one
  `api_contract` test to fail.

### D13. Add `CHANGELOG.md` + deprecation policy (D-API)

A `CHANGELOG.md` at the repo root with a Keep-a-Changelog format, and
a one-paragraph deprecation policy: how long a deprecated symbol
stays before removal, how `DeprecationWarning` is raised, where it is
documented.

- Files: new `CHANGELOG.md` (start with the OSS release as v0.1.0),
  `docs/api/stability.md` (one page summarizing §3.2 + the deprecation
  policy), cross-link from `README.md`.
- Acceptance: `CHANGELOG.md` parses with a Keep-a-Changelog linter;
  `docs/api/stability.md` lists every symbol in
  `building2building/__init__.py.__all__`.

### D14. Document the four benchmark problems on the docs site (D-API)

Each problem (cross-domain, dynamics adaptation, goal adaptation,
action-space transfer) gets one docs page that explains: the train/test
split, the held-constant axes, the metric, the relevant baseline, and
one minimal code example using `new_make_env` + a benchmark class.

- Files: `docs/benchmarks/{cross_domain,dynamics_adaptation,goal_adaptation,action_space_transfer}.md`,
  `mkdocs.yml` nav update.
- Acceptance: `mkdocs build --strict` passes; each page renders an
  end-to-end runnable code block; cross-linked from the paper PDF
  link in `README.md`.

### D1. Add MIT LICENSE

- Files: new `LICENSE` (MIT, year + author lines populated);
  `pyproject.toml` (`license` field); `README.md` footer.
- Acceptance: `LICENSE` file exists; `pyproject.toml` parses; README
  cites it.

### D2. Delete the legacy reward family (PRECEDES C1)
val_dynamics_adaptation.py` — fix
    `compute_normalized_score` argument order; replace hardcoded
    `PadObservation(target_size=20)` with a value read from the
    training run's metadata.
  - `baselines/plotting/plot_ppo_specialist.py` — verified column
    name match.
- Acceptance: each bug has a regression test under `tests/quick/`;
  `pytest -m quick` green.
Remove `task1`–`task5`, `BarrierRewardConfig`, `BaseRewardConfig`,
the un-normalized `DeadbandRewardConfig`, and corresponding reward
classes in `simulator/rewards.py` (keep only `NormalizedDeadbandReward`
and the shared `_deadband_components` helper).

- Files: `building2building/types.py`, `building2building/config/tasks.py`,
  `building2building/simulator/rewards.py`, `building2building/__init__.py`
  (exports), `building2building/api/__init__.py` (default `task=`
  argument), `baselines/configs/reward/task[1-5].yaml` (delete),
  `baselines/configs/experiment/*.yaml` (replace task references),
  `tests/quick/test_task_presets.py`, `tests/quick/test_*` (drop
  legacy-only tests), `README.md`, `docs/`, `tutorials/*` (update
  examples).
- Acceptance: `rg -F 'task1' building2building/ baselines/ tests/
  tutorials/ docs/` returns nothing relevant; `rg 'BarrierReward|BaseReward'
  building2building/ baselines/ tests/` returns nothing;
  `pytest -m quick` is green.
- Reference: `notes.md` § "Status snapshot" and § "Active decisions".

### ~~D3. Fix the documented eval bugs~~ ✓

Four small bugs in the README "Known Issues" section.

- Files:
  - `baselines/eval_ppo.py` — fix `compute_normalized_score`
    argument order; fix the model path lookup to match
    `models/<type>/<task>/ppo_<id>.zip`; fix CSV column name
    (`reward` → `reward_mean`) so `plotting/plot_ppo_specialist.py`
    works.
  - `baselines/eval_dynamics_adaptation.py` — fix
    `compute_normalized_score` argument order; replace hardcoded
    `PadObservation(target_size=20)` with a value read from the
    training run's metadata.
  - `baselines/plotting/plot_ppo_specialist.py` — verified column
    name match.
- Acceptance: each bug has a regression test under `tests/quick/`;
  `pytest -m quick` green.

### ~~D4. Drop `baselines/requirements.txt`; update install docs~~ ✓

It is incomplete and shadows `pyproject.toml[training]`.

- Files: delete `baselines/requirements.txt`; update
  `baselines/README.md` and root `README.md` to point at
  `pip install -e ".[training]"`.
- Acceptance: `requirements.txt` is gone; install instructions point
  exclusively at `pyproject.toml`.

### ~~D5. Write `REPRODUCING.md`~~ ✓

A single document mapping every paper figure / table to the exact
command (Hydra invocation, expected outputs, plotting command).

- Files: new `REPRODUCING.md` at repo root; cross-linked from
  `README.md` (Documentation section) and `design_doc.md`
  (§2 Reproducibility principle).
- Coverage: Phase-C deliverables (`baseline_returns.csv`, PPO
  specialist, **SAC specialist** (camera-ready addition),
  dynamics adaptation, cross-domain Amorpheus); plus extended
  scope — Phase-B calibration / B2 SAC ablation / B3 PPO `target_kl`,
  reactive-controller Optuna tuning, PPO CHS hyperparameter sweep,
  smoke-test tier. Each entry has the exact Hydra one-/two-liner
  *and* the plotting command, plus expected artefact paths.
- Camera-ready note: commands target the normalized 3 × 3 task
  family (`task_{const,occ,rand}_{w0,wmed,whigh}`); `REPRODUCING.md`
  flags every spot where a config still defaults to a legacy
  `task1`–`task5` reference and will be updated as part of D2 / C1.
- Follow-up tracked: `plot_ppo_specialist.py` needs a third
  bar-group for SAC (or a duplicate `plot_specialists.py`) before
  the camera-ready figures can render PPO + SAC side-by-side.

### D6. Add `baselines/` smoke tests

- Files: new `tests/quick/test_train_ppo_smoke.py`,
  `tests/quick/test_train_sac_smoke.py` (extend existing),
  `tests/quick/test_eval_ppo.py` covering D3 fixes,
  `tests/quick/test_run_reactive_control_smoke.py`.
- Acceptance: all new tests run in < 60 s total; `pytest -m quick`
  green.

### D6.5. Audit and triage the test suite (precedes D7)

**Start with a user brainstorm** before touching any files. The agent
must present the candidate changes (keep / add / remove / fix) and get
explicit sign-off before making any edits.

**Motivation.** `pytest tests/quick/` currently shows 27 failures, all
pre-existing and unrelated to recent code changes. Four root causes:

1. **Missing training extras** (16 tests) — `torch`, `stable_baselines3`,
   `hydra`, `matplotlib` not installed under the base `[test]` extra.
   Tests in `test_train_sac_smoke.py` and one in `test_rl_wrappers.py`
   hard-import training-only modules at collection time.
2. **Dataset not downloaded** (8 tests) — `test_climate_zones.py`,
   several `test_rl_wrappers.py`, `test_rollout.py` call `new_make_env`
   or read the HuggingFace parquet registry, which is a 0-byte
   placeholder until the dataset is fetched.
3. **Missing fixture directories** (2 tests) — `test_data_registry.py`
   expects `tests/fixtures/fake_dataset/{OfficeSmall,Warehouse}/` which
   don't exist in the repo.
4. **Deleted preset still in tests** (4 tests) — `test_task_presets.py`
   checks for `task3_legacy` which was removed from `TASK_PRESETS` as
   part of the legacy reward cleanup (see D2).

**Brainstorm questions to resolve before acting:**

- Root cause 1: should training-only tests be guarded with
  `pytest.importorskip("torch")` / a `training` mark, or moved to a
  separate `tests/training/` tier that only runs under `--extra
  training`?
- Root cause 2: should those tests be gated on dataset availability
  (e.g. `pytest.mark.requires_dataset`), mocked out, or kept as-is as
  long tests that only run in full CI?
- Root cause 3: should the fake dataset fixtures be created from
  scratch, or should those tests be reworked to use `tmp_path` +
  `mock`?
- Root cause 4: should the `task3_legacy` tests be deleted outright
  (since D2 removes the preset), or kept as `xfail` until D2 lands?
- Are there other tests that are passing but testing code that has since
  been deleted or superseded?
- After the triage, `pytest -m quick` (no extras, no dataset) must be
  green.

- Files: `tests/quick/`, `tests/conftest.py`, possibly new
  `tests/fixtures/`.
- Acceptance: `pytest -m quick` exits 0 under `uv sync --extra test`
  (no training extras, no dataset download); the brainstorm document is
  preserved as a comment in this TODO item or a linked PR description.

### D7. Add CI

A minimal GitHub Actions workflow.

- Files: `.github/workflows/test.yml` running `pip install -e ".[test]"`
  and `pytest -m quick` on push and PR; matrix over Python 3.10 + 3.11.
- Acceptance: workflow file is valid (`actionlint` or equivalent);
  passes when triggered.

### D8. Update `README.md` "Known Issues" section

After D3 lands.

- Files: `README.md`.
- Acceptance: items fixed in D3 are removed; remaining limitations
  (e.g. legacy SAC stability if Phase B is incomplete at release
  time) are listed honestly.

### D9. Documentation pass for the new reward family

Three concerns roll into this single docs pass:

1. **Migrate every tutorial and doc snippet off the legacy task
   family** (`task1`–`task5`) onto a normalized preset.
2. **Reconcile the `tutorials/*.py` ↔ `docs/tutorials/*.md`
   duplication.** The two directories are not duplicates — they are
   1:1 paired (runnable Python script ↔ MkDocs prose) — but the code
   in the `.md` files is hand-copied from the `.py` files and has
   already drifted (e.g. `quick-tour.md` Step 5 uses `obs` where
   `quick_tour.py` uses `obs2`; `train-ppo.md` is missing the
   train/eval split that `train_ppo_specialist.py` shows). Pick one
   of:
   - **D-lite (recommended):** keep both directories; convert each
     `.md` to thin prose + a link to the `.py` source. No new
     dependencies.
   - **D-full:** use `pymdownx.snippets` to inject `.py` content into
     the `.md` via `--8<--` includes. Single source of truth.
   - **Collapse:** delete one side. Loses either runnability
     (`docs/tutorials/*.md` only) or rendered docs
     (`tutorials/*.py` only); both are worse than D-lite.
3. **Decide on the task preset names** (open question in
   `notes.md`). If renaming (`task_const_w0` → `comfort_only_const`,
   etc.), do it in this commit so the docs land with their final
   names.

- Files: `docs/api/types.md`, `docs/api/scoring.md`,
  `docs/baselines/*.md`, `docs/getting-started.md`,
  `docs/tutorials/*.md`, `tutorials/*.py`, plus `mkdocs.yml` if
  D-full and a new extension is added.
- Acceptance: `mkdocs build --strict` passes; `tutorials/quick_tour.py`
  and `tutorials/train_ppo_specialist.py` run end-to-end on a
  smoke-sized config; `rg 'task1\|task2\|task3\|task4\|task5'
  docs/ tutorials/` returns nothing; no code block in
  `docs/tutorials/*.md` duplicates code that also lives in
  `tutorials/*.py`.

### D10. Final cleanup

- Files: confirm `.gitignore` covers any remaining ignored working
  dirs (the 2026-05-12 cleanup deleted `scrap/`, `staging/`,
  `refactoring/`, `plans/`, `wandb/`, `site/`, `logs/`; the
  `.gitignore` entries are retained as tripwires); run `black .`;
  run `pyright building2building baselines`; review and resolve any
  remaining `WIP` / `TODO` comments in code.
- Acceptance: `git status` after a clean checkout is empty for
  ignored dirs; `black --check .` and `pyright` are clean (or any
  remaining issues are documented in `notes.md`).

### D11. Delete `summaries/`

After `notes.md` has been validated as a complete substitute and
nothing in `summaries/` is still referenced.

- Files: `git rm -r summaries/`; remove any leftover references in
  `design_doc.md` / `notes.md`.
- Acceptance: `rg summaries/` returns nothing; user has confirmed
  `notes.md` is sufficient.

---

## Phase T — Test suite cleanup and core coverage

The test suite has accreted around bug fixes rather than around the
project's stated stability priorities (the building processing
pipeline and the public API). This phase prunes the dead weight and
fills the highest-risk gaps. The umbrella goal is: **after Phase T,
`pytest -m quick` on a fresh checkout should give a contributor
genuine confidence that the pipeline and `new_make_env` still
work**, not just that some legacy regression hasn't reverted.

Reference: `docs/about/testing.md` for the current per-file inventory
and gap analysis.

**One logical change, one commit.** Each item below is sized to be a
single reviewable commit. A few items legitimately fan out (T0, T4,
T11, T24c, T27) — those declare their sub-commits explicitly. The
underlying rule is that every commit lands as a coherent, bisectable
unit, not that every item is exactly one commit.

### Tier definitions for Phase T

Phase T redefines `quick` / `long` along the **rollout length** axis,
not the **EnergyPlus availability** axis. EnergyPlus is a hard
dependency of this project (`tests/conftest.py` already wires it up
once at collection time); pretending otherwise is what produced the
`MockEnv`-everywhere problem in the first place.

| Marker    | Meaning                                                                       |
| --------- | ----------------------------------------------------------------------------- |
| `quick`   | No rollout, or a short rollout (≤ ~20 steps). EnergyPlus and the cached HF dataset are allowed. Aim for each test under a few seconds. |
| `long`    | Multi-day rollouts (hundreds to tens of thousands of steps), or per-cycle leak/lifecycle iteration. Gated on `B2B_RUN_LONG_TESTS=1`. |
| `release` | Dataset / artifact integrity checks against the published `vtaboga/building2building_dataset`. Not run on every push; lives in `tests/release/`. Wired into CI in deferred item TZ1. |

Consequences for Phase T:

- The headline gate "no EnergyPlus, no network" for `quick` is gone.
  `quick` tests may call `new_make_env` against a committed minimal
  fixture and run a 10–20-step rollout. EnergyPlus startup (~1–3 s)
  is acceptable.
- The "long" marker no longer means "needs EnergyPlus" — it means
  "needs a multi-day rollout to be meaningful" (leaks across cycles,
  seasonal-mean assertions, full-year integrity).
- `tests/release/` is new in this phase; T15.0 registers it.

### Test design principles for Phase T

These principles govern every item in Phase T (and should govern
test PRs after the phase closes too). They exist because a test
that lives in its own parallel universe — hand-rolled mocks, fake
config dicts, `MagicMock(spec=...)` stand-ins for B2B objects —
silently drifts as the production code evolves. The real code
changes shape, the mock keeps passing, and the test becomes
false reassurance.

1. **Exercise the real code path.** Build envs through real
   `new_make_env` / `make_env_from_config` against a committed
   fixture; do not build a `MockEnv` that pretends to be a B2B env.
   Construct configs (`TaskConfig`, `EnvBuildConfig`,
   `BuildingInfo`, `BuildingConfig`) via their real
   constructors / `from_dict` / `from_json` methods, not via
   `MagicMock(spec=...)` or hand-rolled dicts.
2. **Mock only at external I/O boundaries.** The HuggingFace
   network call is the only legitimate mock target inside Phase T —
   and even then, the preferred pattern is to point the registry at
   the committed minimal fixture via the shared `fixture_registry`
   helper (T0), not to monkeypatch ad-hoc. EnergyPlus itself is not
   mocked.
3. **Minimal, shared, real fixtures.** Prefer the committed
   minimal-building matrix (T0: one fixture per HVAC type) +
   one committed equipment.json per HVAC type (T9) reused across
   the whole quick suite, over per-test hand-rolled fixtures.
   Every fixture is documented in a `README.md` next to it so its
   provenance is reproducible.
4. **Mock surface is itself a code smell.** If a test needs more
   than ~5 lines of `monkeypatch` / `MagicMock` setup, that is a
   signal the test is at the wrong layer. Extract a small helper
   from production code and test that directly (see T4 for the
   pattern: T4a extracts a helper, T4b tests it).
5. **Wrappers and other generic `gym.Env` consumers may use a
   `MockEnv`** — those are general by design. But where the
   wrapper interacts with B2B-specific metadata (e.g.
   `PadObservation` reading `observation_names`), the test must
   *also* gain a companion test against the real minimal-building
   fixture, so a wrapper that works on `MockEnv` but breaks on the
   real `metadata` / `observation_space` shape is caught.

These are guidelines, not absolute rules — the goal is fewer
sources of false positives, not maximum integration-test purity.

### T0. Minimal-building fixture matrix + shared registry helper

The single biggest unlock. Today, almost every "real" test either
mocks too much (and tests nothing useful) or hits the HuggingFace
dataset. A committed minimal fixture matrix plus a shared
registry-bypass fixture removes that dilemma without expanding
`new_make_env`'s public signature.

**HVAC coverage rule:** every advertised HVAC system type
(`VAV`, `Unitary`, `HeatingOnly`) gets its own committed minimal
building. This is the only way the quick suite ever exercises the
`make_controllable` / equipment-dispatch path for all three types
without depending on the real registry.

T0 fans out into three commits, each independently reviewable:

#### T0a. Fixture files and provenance

Commit one fixture directory per HVAC type under
`tests/fixtures/`:

- `minimal_vav/`        — smallest multi-zone building with VAV HVAC
  (2 zones, 2 actuators). Also serves as the multi-zone variant
  used by T17 (no second 2-zone fixture is needed).
- `minimal_unitary/`    — smallest single-zone Unitary building.
- `minimal_heating_only/` — smallest single-zone HeatingOnly
  building, with at least one `Schedule:File` reference (to
  exercise T8's relative-path rewriting on the SFH-style code
  path).

Each directory contains `building.epjson`, `equipment.json`,
`weather.epw`, and a `README.md`. The `building.epjson` is a
post-conversion artifact, *not* an IDF — T5 owns the IDF → epJSON
path.

Each `README.md` documents: which real dataset building it was
carved from, what was deleted (zones, equipment, schedules),
pinned values for `area` / `warmup_phases` / `hvac_actuators`
count (T7 asserts against these), and the **measured per-cycle
cost** of `new_make_env().reset().close()` against this fixture on
the dev box (so future contributors know whether a slow CI is
because of the fixture or because of their change).

**Run period:** target one-week (Jan 1 – Jan 7) baked into each
`building.epjson`. EnergyPlus's warmup is independent of the
reported run period, so one week is fine in principle, but **T0a's
first step is to verify experimentally** that each fixture
produces ≥10 reported timesteps at `timesteps_per_hour=4`. If any
fixture fails that, fall back to one month *for that fixture
only* and document the reason in its README. Do not pre-emptively
bake one month into all three.

Total committed size budget: ~1.5 MB across three epJSONs. If any
single fixture exceeds 800 KB, carve more aggressively before
committing.

- Files: `tests/fixtures/{minimal_vav,minimal_unitary,minimal_heating_only}/*` (new).
- Acceptance: each fixture round-trips through `new_make_env`
  against `fixture_registry` (added in T0b) and reaches `reset()`
  in under 5 s on the dev box; the README's measured cost matches
  within 50 %.

#### T0b. Shared registry helper

- `fixture_registry` fixture in `tests/conftest.py`, parametrized
  by HVAC type (default: `minimal_vav`). Returns a stub registry
  whose `get_building_by_index` / `get_building_by_id` return a
  real `BuildingInfo` built from the requested fixture (not a
  `MagicMock`). Tests use it via `monkeypatch.setattr(
  building2building.data.registry, "get_registry",
  lambda: fixture_registry)`. The pattern is established by the
  existing `_StubRegistry` in
  `tests/quick/test_api_mode_default.py` — promote that to a
  shared fixture and extend it for HVAC-type parametrization.
- `minimal_building_dir(hvac_type)` fixture in `tests/conftest.py`
  returning the `Path` to the requested fixture directory.

Rationale for not extending `new_make_env`: `new_make_env` is the
public, paper-cited entry point. Adding a `building_dir=` knob just
to make tests easier expands the public surface for a test
convenience and invites users to skip the registry. The monkeypatch
pattern is ~3 lines per call site and touches no production code.

- Files: `tests/conftest.py`.
- Acceptance: `pytest tests/conftest.py --collect-only` lists both
  fixtures; a smoke test (added in T10) parametrizes over all
  three HVAC types and passes. No `building_dir=` kwarg on
  `new_make_env`.

#### T0c. Migrate the existing stub in `test_api_mode_default.py`

T0a + T0b leave `tests/quick/test_api_mode_default.py` still using
its own `_StubInfo` / `_StubRegistry` / `_CapturedConfig` machinery.
Replace those with the shared `fixture_registry`. This commit is
small (~30 LOC delete, ~5 LOC change) and **must land before T4b**
otherwise T4b is doing this migration plus the helper rewrite in
one go.

- Files: `tests/quick/test_api_mode_default.py`.
- Acceptance: file loses the `_Stub*` classes; existing assertions
  still pass.

### T1. Delete tautological tests

Goal: shrink the suite, but **no contract is silently dropped**.
Each deletion below either has no contract attached, or names the
test that absorbs the contract.

- **Delete** `tests/quick/test_new_api.py`. Only asserts re-export
  existence; covered transitively by every test that imports
  `building2building.<symbol>` and uses it.
- **Collapse**
  `test_gym_registration.py::test_registration_on_import` into
  `test_all_building_types_registered` (one assertion suffices for
  "import has the side-effect").
- **Delete**
  `test_eval_bugs.py::TestDynamicsAdaptationMetadataRoundTrip::{test_train_writes_metadata_json, test_eval_reads_pad_obs_size_from_metadata}`
  — these only round-trip a dict through `json`. No B2B contract.
- **Preserve the contract**
  `test_eval_bugs.py::TestDynamicsAdaptationMetadataRoundTrip::test_metadata_missing_raises_without_cli_flag`
  exists in a clumsy form (raises and catches `FileNotFoundError`
  in the same line). The contract — "eval fails loudly when
  metadata.json is missing" — is real. Rewrite it in T27 step 3
  against the actual `eval_dynamics_adaptation` entry point and
  move into `tests/quick/test_eval_path_layout.py`. **Do not
  delete it in T1.**
- **Delete**
  `test_eval_bugs.py::TestEvalPpoCsvRewardMeanColumn::test_csv_fieldnames_contain_reward_mean`.
  The contract ("eval_ppo writes `reward_mean`, not `reward`") is
  preserved by the sibling `test_eval_result_has_reward_mean_not_reward`
  in the same class (which T1 keeps).
- **Keep**
  `test_reward_normalizers.py::test_default_path_constant_points_inside_package`
  — it pins that the shipped YAML lives at the expected package
  path, which silently breaks if `building2building/data/` is
  reorganized. Cost: one line. T27 step 8 re-evaluates after
  `data/` stabilises post-release.
- Acceptance: `pytest -m quick` still green; line count down;
  every deleted test is either contract-free or its contract has
  a named survivor.

### T2. Consolidate duplicates

For each duplicate, name the winner explicitly **and** name every
assertion in the loser that the winner doesn't already cover (those
get merged into the winner before deletion).

- **`tests/test_building_param_clipping.py` → `tests/test_wrappers.py`.**
  Both test the same `AugmentObservationWithBuildingParams`
  clipping invariant on a `MockEnv`. Move the one assertion into
  `TestAugmentObservationWithBuildingParams` as
  `test_normalized_params_clipped_when_out_of_range`. T24c rewrites
  the surrounding test class anyway, so this just keeps the
  invariant alive in the meantime.
- **Seasonal `from_dict` roundtrip:** keep
  `tests/quick/test_seasonal_target.py::TestZoneTargetTemperatureConfigSeasonal::test_from_dict_seasonal`
  (it lives with the runtime tests for the same policy, which is
  the more useful neighbour). In
  `tests/quick/test_types.py::TestZoneTargetTemperatureConfigExtended`,
  delete **only** `test_from_dict_seasonal_roundtrip`. **Keep**
  `test_from_dict_unknown_policy_raises` and
  `test_from_dict_seasonal_unknown_season_key_raises` — both
  encode unique error-path contracts (unknown policy string,
  unknown season key) that are not duplicated elsewhere.
- **`test_task_presets.py::test_legacy_task1_unchanged` vs.
  `test_task1_deadband_low_energy_weight`:** keep
  `test_legacy_task1_unchanged` (its docstring spells out the
  reproducibility contract — "existing PPO results /
  baseline_returns rows remain reproducible"). Before deleting
  `test_task1_deadband_low_energy_weight`, **merge its missing
  assertion** (`assert p.unoccupied_policy == "fixed"`) into the
  survivor, and conversely keep `test_legacy_task1_unchanged`'s
  unique assertion (`assert p.reward.dT == 1.0`). Net effect: one
  surviving test, no assertions lost.
- Acceptance: each surviving test is reachable from exactly one
  file; every assertion present before T2 is still reachable
  after T2 (run `pytest -m quick` and grep the diff for `assert`
  before merging); `pytest -m quick` green.

### T3. Audit `tests/long/` against the new rollout-length definition

Under the new tier definition, `long` means *multi-day rollout*, not
*needs EnergyPlus*. Walk every file in `tests/long/` and confirm it
actually justifies the marker. Inverse of the old T3 — we no longer
move `quick/` tests out, we audit `long/` tests in.

Concrete cases to verify:

- `tests/long/test_pipeline_single_zone_houses.py` runs a single
  `reset()` + one `step()` (max_episode_steps=8). It is **not**
  long under the new definition. Marked for deletion in T27 (its
  contract is fully covered by T5's SFH pipeline matrix and T20's
  benchmark smoke test).
- `tests/long/test_rescale_action.py`, `test_observation_dimension.py`
  — verify whether they truly need multi-day rollouts or just an
  env build + a handful of steps. If the latter, move to `quick/`
  in T27.
- `tests/long/test_env_leak.py`, `test_env_lifecycle.py`,
  `test_seasonal_unoccupied.py`, `test_occupancy_observation.py`,
  `test_random_schedule_rollout.py`,
  `test_all_buildings_env_smoke.py` — these genuinely need many
  steps or many envs. Keep `long`.
- Acceptance: every surviving `tests/long/` file has a one-line
  module docstring justifying *why* it needs to be long. Any file
  that can't justify the marker either moves to `quick/` or gets
  deleted in T27.

### T4. Rewrite over-mocked tests against the layer they actually mean to test

T4 is a small *cluster* of independent rewrites. T4a is the only
production-code change; T4b–T4d are pure test rewrites. Each is its
own commit.

#### T4a. Extract two helpers from `new_make_env`

Pure refactor, no behaviour change. The inline block in
`new_make_env` (`building2building/api/__init__.py`, roughly lines
252–353) does **two** distinct things:

1. **TaskConfig resolution** — preset lookup, effective-mode
   override, default `ZoneTargetTemperatureConfig` build,
   `RandomScheduleConfig` build, `TaskConfig` build.
2. **Effective reward resolution** — `effective_reward = reward if
   reward is not None else preset.reward`, plus the
   `NormalizedDeadbandRewardConfig` autofill path that looks up
   `(tau_T, tau_E)` from `reward_normalizers.yaml` using the
   building's `(building_type, climate_zone)` bucket.

(2) is the more bug-prone branch — it can `KeyError` at env-build
time on a stale YAML, and the existing tests do not exercise the
autofill. Extracting both in one commit means T4b can test each
independently.

Extract into two module-level helpers:

```python
def _resolve_task_config(
    *,
    preset: TaskPreset,
    run_period_cfg: RunPeriodConfig,
    timesteps_per_hour: int,
    target_temperature_mode: str | None,
    random_schedule_seed: int | None,
    building_type: BuildingType,
) -> TaskConfig: ...

def _resolve_effective_reward(
    *,
    preset: TaskPreset,
    reward_override: RewardConfig | None,
    building_type: BuildingType,
    building_id: str,
    run_period: str,
    normalizer_path: Path | None,
) -> RewardConfig: ...
```

Existing tests (`test_api_mode_default.py`, all env-build tests)
must keep passing unchanged. This commit unblocks T4b.

- Files: `building2building/api/__init__.py`.
- Acceptance: `pytest -m quick` green; the inline block in
  `new_make_env` is reduced to one call to each helper; both
  helpers are importable from `building2building.api` (private
  underscore prefix, but importable for T4b's tests).

#### T4b. Rewrite `test_api_mode_default.py` against the new helpers

T0c already removed `_StubInfo` / `_StubRegistry` / `_CapturedConfig`
from this file. T4b drops the remaining `create_simulator`
monkey-patch and tests the helpers directly:

- For each paper preset (`task1`–`task5`) and each normalized
  preset, call `_resolve_task_config` with real `TaskPreset`
  instances from `TASK_PRESETS` and assert on the returned
  `TaskConfig` (mode, default zone target, random schedule).
- For `_resolve_effective_reward`: parametrize over (a) a normal
  preset with `reward_override=None` (returns `preset.reward`
  unchanged), (b) a normalized preset (returns a *filled*
  `NormalizedDeadbandRewardConfig`), (c) an explicit reward
  override (returns the override). Use `fixture_registry` (T0b)
  to provide the building's climate zone for the autofill path,
  or commit a tiny `tests/fixtures/reward_normalizers_fixture.yaml`
  pointing `normalizer_path` at a one-bucket file.
- Keep one quick integration test that builds a full env via
  `new_make_env` against `fixture_registry` and asserts
  `env.unwrapped.task_config.target_temperature_mode` end-to-end.

- Files: `tests/quick/test_api_mode_default.py`.
- Acceptance: file shrinks by ~50% relative to its post-T0c size;
  total mock surface in the file is under 5 lines; the
  reward-autofill `KeyError`-on-stale-YAML case is now covered
  (use a one-bucket fixture YAML and request a building outside
  the bucket; assert `KeyError`).

#### T4c. `test_climate_zones.py::TestMetadataColumnGuard`

Replace the `pd.read_parquet` / `download_metadata` double
monkey-patch with a unit test of a tiny `_validate_metadata(df)`
helper extracted from `BuildingRegistry._ensure_loaded`. The
helper takes a `DataFrame`, raises the same `RuntimeError` if
`climate_zone` is missing, returns `None` otherwise. The test
constructs a `DataFrame`, calls the helper, asserts the raise.

- Files: `building2building/data/registry.py` (extract helper);
  `tests/quick/test_climate_zones.py` (rewrite test).
- Acceptance: zero monkey-patches in the rewritten test; the
  `_ensure_loaded` path keeps its existing behaviour.

#### T4d. `test_task_presets.py::test_factory_lazy_imports_loader`

The current grep-the-source test is correct in intent but too
narrow: it only catches a direct top-level import in `tasks.py`.
A reward-normalizer import added to any other file that
`config.tasks` transitively imports at top level would still
trigger dataset I/O at `import building2building.config.tasks`
time, but the grep test wouldn't notice.

Rewrite as a runtime contract — "after `import
building2building.config.tasks`, `reward_normalizers` is not in
`sys.modules`" — executed in a **fresh subprocess** so the result
isn't polluted by the parent pytest process's module cache:

```python
import subprocess, sys, textwrap

def test_factory_lazy_imports_loader() -> None:
    code = textwrap.dedent("""
        import sys
        import building2building.config.tasks  # noqa: F401
        assert (
            "building2building.data.reward_normalizers" not in sys.modules
        ), sorted(m for m in sys.modules if m.startswith("building2building"))
    """)
    subprocess.run([sys.executable, "-c", code], check=True)
```

The subprocess is necessary: pytest itself imports
`building2building` (via `conftest.py`'s `setup_energyplus_path`),
which may transitively pull `reward_normalizers` into the parent
process's `sys.modules` cache long before the test runs. Popping
from `sys.modules` and re-importing in the same process does
*not* re-execute the module body — cached child symbols keep the
original module alive — so an in-process version of this test
would have false negatives.

- Files: `tests/quick/test_task_presets.py`.
- Acceptance: the test fails if anyone adds a top-level import of
  `reward_normalizers` to `tasks.py` *or* to any module
  transitively imported at file scope. The subprocess overhead
  (~200 ms for Python startup) is acceptable for a single test.

#### T4e. Delete `test_wrappers.py::test_pad_raises_error_if_too_many_zones`

The current test's own comment admits the code path hits the size
check first, not the zone-specific one — i.e. the test name is a
lie. Delete it. T24b adds a proper zone-specific error test
against real `observation_names` metadata (the assertion this
test was *trying* to make), and
`test_pad_raises_error_if_obs_too_large` already covers the size
check that this test actually hits.

T4e must land **after** T24b so that the zone-specific contract
is never uncovered between commits. If T4e is reviewed first and
T24b is not yet merged, hold T4e.

- Files: `tests/test_wrappers.py`.
- Acceptance: `pytest -m quick` green; T24b's zone-specific test
  is present in the same merge train.

### T5. Pipeline: end-to-end `prepare_building` on a fixture IDF

The pipeline (`building2building/pipeline/`) is the project's biggest
untested surface and the highest blast radius — pipeline bugs
silently corrupt the dataset.

The fixture matrix **must include one SFH building with a
`Schedule:File` reference**. SFH `Schedule:File` rewriting is the
specific footgun that `_patch_epjson_run_period` exists for (T8
covers the post-conversion patching; T5 covers the pre-conversion
pipeline). Without SFH in T5's matrix, the SFH-specific paths in
`prepare_building` stay implicitly tested by
`tests/long/test_pipeline_single_zone_houses.py` — which T27 deletes.

- Files: new `tests/quick/test_pipeline_prepare_building.py`. Use
  tiny committed IDF fixtures under `tests/fixtures/pipeline_idfs/`
  — at minimum one multi-zone (VAV) and one SFH (with at least one
  `Schedule:File` object pre-conversion). Parametrize the test
  over the matrix. For each, run `prepare_building` through to
  epJSON and assert:
  (a) HVAC + outdoor-air meter objects present,
  (b) `Timestep:NumberOfTimestepsPerHour` matches the input,
  (c) `RunPeriod` spans the full year,
  (d) the result round-trips through `json.load`,
  (e) for SFH: every `Schedule:File` survives the conversion (the
      conversion itself doesn't rewrite paths — that's T8's job —
      but the objects must be present and parseable post-conversion).
- Marker: `quick` under the new tier definition. EnergyPlus is
  invoked for the IDF → epJSON conversion (~1–3 s per case), no
  rollout.
- Acceptance: `pytest tests/quick/test_pipeline_prepare_building.py`
  green; both VAV and SFH cases run.

### T6. Pipeline: `make_controllable` per HVAC system type

`make_controllable` can run against a pre-converted epJSON, so this
is `quick` — no IDF → epJSON step required at test time.

- Files: new `tests/quick/test_pipeline_make_controllable.py`. One
  parametrize per system type (`VAV`, `Unitary`, `HeatingOnly`),
  reusing the three post-conversion epJSONs committed by T0a
  (`tests/fixtures/minimal_{vav,unitary,heating_only}/building.epjson`).
  Assert the returned `ActuatorDescription` list contains the
  expected `(component_type, control_type)` shape and no
  `autosized` entries.
- Acceptance: catches the case where `make_controllable` silently
  emits the wrong actuator set for one of the three system types
  (today, only VAV + Unitary are exercised, and only indirectly
  via the benchmarks tests).

### T7. Pipeline: `extract_discovery_metadata`

- Files: new `tests/quick/test_pipeline_discovery.py`. Parametrize
  over the three T0a fixtures; for each, run discovery and
  assert `area`, `warmup_phases`, and `hvac_actuators` come out
  with the values pinned in
  `tests/fixtures/minimal_{vav,unitary,heating_only}/README.md`.
- Marker: `quick` — discovery runs against a committed
  post-conversion epJSON; EnergyPlus is invoked once per fixture
  for the EDD dump (T7 implicitly verifies that each T0a fixture
  round-trips through EnergyPlus, which is useful free coverage).
- Acceptance: any future change to the discovery parser that
  silently shifts a returned value fails this test for at least
  one HVAC type.

### T8. API: `_patch_epjson_run_period` schedule-file rewriting + RunPeriod patching

The `Schedule:File` relative-path rewriting in
`building2building/api/__init__.py::_patch_epjson_run_period` is a
known footgun (EnergyPlus segfaults without it) with zero coverage.
Pure in-memory test; no EnergyPlus.

- Files: new `tests/quick/test_patch_epjson_run_period.py`. Build a
  tiny epJSON `dict` in-memory and write it to disk in a temp
  source directory. Cover:
  (a) **Relative `Schedule:File` paths get resolved to absolute
      paths rooted at `src_epjson.parent`** when `dst_epjson` lives
      in a different directory. Use `Path(...).resolve()` in the
      expected value to match the production code's behaviour.
  (b) **Absolute `Schedule:File` paths are left alone.**
  (c) **`winter` RunPeriod** dates are rewritten correctly
      (`begin_month`, `begin_day_of_month`, `end_month`,
      `end_day_of_month` match the `RunPeriodConfig`).
  (d) **`summer` RunPeriod** dates ditto.
  (e) **No-RunPeriod source epJSON** — when the input lacks any
      `RunPeriod`, the function fabricates one with hard-coded
      `begin_year=2023`, `apply_weekend_holiday_rule="No"`, etc.
      Lock that contract or, if you'd rather raise, change the
      production code in a sibling commit and update the test.
- Acceptance: quick test; protects the SFH segfault fix and the
  full-year-default fabrication branch.

### T9. Pipeline: `equipment.json` schema round-trip per HVAC type

T0a already commits one `equipment.json` per HVAC type inside each
fixture directory
(`tests/fixtures/minimal_{vav,unitary,heating_only}/equipment.json`).
T9 reuses those — no new fixture files.

- Files: new `tests/quick/test_equipment_schema.py`. Parametrize
  over the three HVAC types; for each, load the T0a fixture's
  `equipment.json` and assert
  `cattrs.structure(json.loads(...), list[AnyEquipment])` succeeds
  and produces objects of the expected concrete type.
- Acceptance: schema drift in `pipeline/actuators.py` is caught by
  a quick test instead of by every `new_make_env` call dying.

### T10. API: `new_make_env` against the minimal fixture matrix

- Files: new `tests/quick/test_new_make_env_minimal.py`. Use the
  T0b `fixture_registry`, parametrized over HVAC type (defaults
  to `minimal_vav` for most knob-coverage cases; HVAC-type-
  specific cases pick the matching fixture). Assert the returned
  env has the expected `observation_space`, `action_space`, and
  `metadata["controlled_zones"]`. Parametrize over the knobs that
  have historically regressed: `target_temperature_mode`,
  `random_schedule_seed`, `reward` override, `run_period`,
  `rescale_action`, `max_episode_steps`. Each parametrize case
  builds the env, asserts on its spaces / metadata, optionally
  runs ≤ 20 steps, and closes.
- Marker: `quick`. EnergyPlus runs once per parametrize case
  (~1–3 s each); no multi-day rollout.
- Acceptance: covers every public `new_make_env` knob through the
  T0b registry path; no HuggingFace cache is touched. At least
  one knob-coverage case parametrizes over HVAC type to keep all
  three fixtures exercised through the full env-build path (not
  just the per-pipeline tests in T5/T6/T7).

### T11. API: `new_make_env` cleanup contract for the epjson staging dir

`weakref.finalize(env, _shutil.rmtree, _epjson_staging_dir, True)`
in `new_make_env` is critical when `run_period != "full_year"`.
Never tested. `del env; gc.collect()` is **not** a reliable
trigger across `TimeLimit` / `RescaleAction` wrappers — the
finalizer fires only when the *innermost* env loses all
references, and wrapper references can outlive the test scope
unpredictably.

Test design: monkey-patch `weakref.finalize` in the
`building2building.api` namespace with a recording wrapper that
captures `(target, callback, args, kwargs)` for every call, then
invoke the captured callback directly:

```python
import weakref
from building2building import api as api_mod

calls: list[tuple] = []

class _RecordingFinalize:
    def __init__(self, target, callback, *args, **kwargs):
        calls.append((target, callback, args, kwargs))
        self._inner = weakref.finalize(target, callback, *args, **kwargs)

    def __call__(self):
        return self._inner()

monkeypatch.setattr(api_mod.weakref, "finalize", _RecordingFinalize)

env = api_mod.new_make_env(
    "OfficeSmall",
    task="task1",
    run_period="winter",       # triggers the staging-dir branch
)
# Exactly one finalize was registered, for the staging dir.
assert len(calls) == 1
target, callback, args, kwargs = calls[0]
import shutil
assert callback is shutil.rmtree
staging = args[0]
assert staging.is_dir()                # exists before invocation
calls[0]                                # invoke the captured callable
# fire the inner finalize directly via the recording wrapper:
env._test_finalize_handle = None       # only needed if you stored one
# call the original finalize handle:
_RecordingFinalize(target, callback, *args, **kwargs).__call__()
assert not staging.exists()            # cleanup ran
env.close()
```

(Sketch — flesh out in the test file. The key points: the recording
wrapper preserves the real `weakref.finalize` contract; we invoke
the captured `(callback, args)` synchronously to verify the
side effect; we never rely on GC.)

This requires no production-code change — `weakref` is already
imported at module scope in `api/__init__.py` (line 12), so
`monkeypatch.setattr(api_mod.weakref, "finalize", ...)` reaches
the right call site. If a future refactor moves the `import
weakref` inside `new_make_env`, the patch target moves with it
and the test docstring must be updated. (Add an `assert
hasattr(api_mod, "weakref")` to catch that breakage explicitly.)

- Files: new `tests/quick/test_new_make_env_cleanup.py`.
- Marker: `quick` — env build + immediate cleanup, no rollout.
- Acceptance: deterministic, no GC race. The test fails if the
  `weakref.finalize` call in `new_make_env` is dropped, its
  callback is wrong (not `shutil.rmtree`), or its args are wrong
  (not the staging dir).

### T12. API: `Trajectory.from_npz` round-trip with real nested types

- Files: extend
  `tests/quick/test_rollout.py::TestTrajectoryRoundTrip` with one
  case that populates `building_info` (a real `BuildingInfo` built
  from the `fake_metadata` fixture or `fixture_registry`) and
  `task_config` (a real `TaskConfig` built from a preset via
  `_resolve_task_config` from T4a). Assert these survive the
  round-trip.
- Acceptance: the only existing round-trip case sets both to
  `None`, so the serialization path for these nested dataclasses is
  currently untested.

### T13. Scoring: real CSV loader

The scoring CSV has two failure modes worth pinning: schema drift
(column rename) and dataset drift (a `(building_type, task)` row
the paper grid expects is missing). Today both surface as opaque
errors deep inside `b2b.compute_normalized_score(...)`.

- Files: new `tests/quick/test_scoring_csv.py`. Point
  `scoring._load_cache` at the committed
  `tests/fixtures/baseline_returns_fixture.csv`. Cover:
  (a) **Happy path** — the loaded cache has the expected keys
      and values for the rows in the fixture.
  (b) **Column rename** — write a temp variant of the fixture
      with `reward_mean` → `reward`; assert
      `compute_normalized_score` (or whichever public entry
      point first touches the renamed column) raises a clear
      error mentioning the column name.
  (c) **Missing row** — call `compute_normalized_score` for a
      `(building_type, task)` tuple absent from the fixture;
      assert a clear `KeyError` / `LookupError` that names the
      missing tuple. This is the actual runtime failure mode at
      eval time (renames are rare; missing rows are common after
      a partial baseline re-run).
- Acceptance: column renames and missing rows both surface at
  test time with messages that point at the specific column or
  tuple, not at line numbers inside `scoring.py`.

### T14. Data integrity: splits ⊆ metadata, no train/test overlap

T14, T15, T16 are **dataset validation**, not code tests. They live
under `tests/release/` (new tier — see T15.0), not `tests/long/`.
Contributors debugging wrappers should not pay for a full
HuggingFace download when they run `B2B_RUN_LONG_TESTS=1`.

- Files: new `tests/release/test_data_integrity.py`. Against the
  real registry (`get_registry()`), assert:
  (a) every building ID in `splits.json` exists in
      `metadata.parquet`,
  (b) `train ∩ test == ∅` for every building type,
  (c) `test_small ⊆ test` for every building type.
- Marker: `release`.
- Acceptance: one run on a release candidate catches dataset-build
  mistakes that would otherwise distort every paper number by a
  small, unattributable amount.

### T15.0. Register the `release` marker + scaffold `tests/release/`

Prerequisite for T14, T15, T16.

- Files: `pyproject.toml` (register the marker under
  `[tool.pytest.ini_options].markers`); `tests/release/__init__.py`
  (empty); `tests/release/README.md` (new — explains the tier,
  lists the tests, and links to deferred item **TZ1** for the CI
  automation); `tests/conftest.py` (no change — the auto-tagger
  only touches `quick` / `long`; `release` tests must opt in
  explicitly with `@pytest.mark.release`); `docs/about/testing.md`
  (T28 will pick up the documentation pass; T15.0 only needs the
  marker registered).
- Acceptance: `pytest --markers | grep release` lists the marker;
  `pytest -m release` collects zero tests until T14 lands; running
  `pytest` without `-m release` does **not** collect the release
  tests (verify with a placeholder test in the directory).

### T15. Data integrity: `reward_normalizers.yaml` covers the metadata

- Files: new `tests/release/test_reward_normalizers_coverage.py`.
  For every `(building_type, climate_zone)` pair present in
  `metadata.parquet`, assert that `reward_normalizers.yaml` has a
  matching bucket. SFH bucketed under `cz0`; multi-zone types
  bucketed under their climate zone.
- Marker: `release`.
- Acceptance: prevents the
  `new_make_env(task="task_occ_wmed", building_id=...)` →
  `KeyError` failure mode at eval time.

### T16. Data integrity: `baseline_returns` covers the paper grid

- Files: new `tests/release/test_baseline_returns_coverage.py`.
  For the `(building_type, task, run_period, building_id)` tuples
  the paper reports, assert each row exists in the scoring CSV.
- Marker: `release`.
- Acceptance: missing rows surface at release time instead of at
  eval time.

### T17. Observation contract: zone-aware padding on real buildings

The "non-zone features at consistent indices" invariant is what
makes multi-building generalization work. Currently tested *once*,
with a `MockEnv`.

T0a already commits a 2-zone `minimal_vav/` fixture and a 1-zone
`minimal_unitary/` (or `minimal_heating_only/`) fixture, so no
new fixture is needed for T17.

- Files: new `tests/quick/test_obs_padding_invariants.py`. Use
  T0b's `fixture_registry` parametrized over a 1-zone and a
  2-zone fixture. Wrap both real envs in
  `PadObservation(target_size=N)` for some N comfortably larger
  than the 2-zone case; assert that for both, the last 7 slots
  are the non-zone features in the same order with non-trivial
  values, and that the padded zone slots in between are zero.
- Marker: `quick`. Two env builds, no rollout.
- Acceptance: catches any future change that puts non-zone features
  at building-dependent indices.

### T18. Observation contract: `observation_names` ordering stability

Every plotting script and reactive controller depends on
`obs_names = env.metadata["observation_names"]`. The ordering is
never asserted.

- Files: new `tests/quick/test_observation_names_stability.py`.
  Snapshot-test the `observation_names` output for the
  `minimal_vav` fixture (the default, multi-zone case is the most
  informative snapshot) under each `target_temperature_mode`.
  Commit expected lists under
  `tests/fixtures/minimal_vav/expected_observation_names_<mode>.json`.
  Fail loudly when the order changes — changing it is fine, but
  should be a deliberate commit that touches the snapshot file,
  not a silent diff. T18 snapshots only `minimal_vav` for now;
  if a future bug surfaces a HeatingOnly-specific ordering issue,
  expand to the other HVAC types then.
- Acceptance: relies on T0.

### T19. Observation contract: `action_space` ↔ `controlled_zones` consistency

- Files: extend the new minimal-building test (T10) with:
  `assert env.action_space.shape[0] == n_agent_actuators`, derived
  from `controlled_zones` and equipment metadata. Don't make it a
  separate file — it's one assertion that belongs alongside T10's
  other space checks.
- Acceptance: catches the case where the action space and the
  declared controlled zones diverge.

### T20. Benchmark behaviour: `make_train_envs` actually returns working envs

The four benchmark classes are tested for their config but not for
behaviour. **T20 stays `long`.** The new tier definition would
allow `quick` (single env build + a handful of steps), but the
benchmarks pull from the real registry by design: each benchmark
fans out to many real buildings even at `n=1` because each
benchmark is a different building/task combination, and pointing
them at `fixture_registry` would hollow out the test (you'd be
asserting that a stub benchmark works on a stub registry, which
is not the contract).

- Files: new `tests/long/test_benchmarks_behaviour.py`.
  For each of `DynamicsAdaptation(difficulty="easy")`,
  `GoalAdaptation()`,
  `CrossDomainGeneralization(difficulty="easy")`,
  `ActionSpaceTransfer(system_type="unitary")`, call
  `.make_train_envs(n=1)` and `.make_test_envs(n=1)`, run one
  `reset` + one `step` per env, assert no crash.
- Marker: `long`. Document the rationale ("real-registry fanout,
  not rollout length") in the file's module docstring so future
  contributors don't try to relabel it `quick`.
- Acceptance: catches behavioural regressions in the benchmark
  factories that the existing config tests cannot see.

### T21. HVAC coverage: `HeatingOnly` system type

`HeatingOnly` is one of the three advertised HVAC types but has no
unit tests. `Unitary` and `VAV` are covered by the
`ActionSpaceTransfer` helpers.

- Files: extend `tests/quick/test_benchmarks.py` with a
  `HeatingOnly` parametrize on `hvac_action_space`. Reuse
  `tests/fixtures/minimal_heating_only/equipment.json` from T0a
  (no new fixture required).
- Acceptance: every advertised HVAC type has at least one
  action-space unit test.

### T22. Leak coverage: multi-zone building

`tests/long/test_env_leak.py` only exercises `SingleFamilyHouse`.
Leak risk is highest on multi-zone buildings.

T22 splits into a measurement step and an execution step so that
"_N=3 takes longer than 5 minutes" doesn't quietly produce a
useless test.

#### T22a. Measure

Time one `OfficeMedium` create/reset/close cycle on the local
interactive node (no commit; record in the T22b commit message
and in the test's module docstring). OfficeMedium per-step cost
can be 3–5× SFH, so this is essential.

#### T22b. Parametrize

Based on T22a's measurement:

- If total `OfficeMedium` runtime at `_N ≥ 5` fits in ~5 minutes:
  parametrize `_BUILDING_TYPE` over `SingleFamilyHouse` (keep
  `_N = 20`) and `OfficeMedium` (pick the largest `_N` that fits
  under 5 minutes; ≥ 5 required for leak signal to dominate
  noise).
- If even `_N = 3` exceeds 5 minutes: **do not ship a `_N = 3`
  leak test**. The signal-to-noise at `_N = 3` is too low to
  catch anything but a catastrophic regression, which the
  existing SFH test already catches. Instead, ship T22b as a
  parallelization refactor (run cycles in subprocesses via
  `multiprocessing` or a `pytest-xdist`-style fan-out) and
  defer the OfficeMedium parametrize to T22c.

- Acceptance: either OfficeMedium leak coverage is in the suite
  with `_N ≥ 5` and total runtime under 5 minutes, or the
  parallelization refactor is committed and a follow-up T22c is
  filed. Do not commit a measurement-free `_N` value.

### T24. Wrapper data-processing correctness

The five wrappers in `building2building/simulator/wrappers.py` and
`building2building/api/rl_wrappers.py` are unevenly tested: existing
tests only hit the `low`/`high` boundaries on a `MockEnv`, never
exercise the `reset()` rebuild paths, never check the
metadata-driven branches, and don't cover
`ResampleBuildingOnResetWrapper` at all. These wrappers sit on the
critical data path between EnergyPlus and the agent — a silent bug
in any of them corrupts every observation or action without
crashing.

One commit per wrapper (T24a–T24e), so a regression in one wrapper
is bisectable to a single sub-task.

#### T24a. `NormalizeObservation`: affine math + reset rebuild

The existing two tests only assert that `low → 0` and `high → 1`,
which is trivially true for any affine map. They miss the actual
arithmetic.

- Files: extend `tests/test_wrappers.py` (or a new
  `tests/quick/test_normalize_observation.py`) with:
  (a) **Mid-range affine check** — for a `Box(low=[10, -5], high=[30, 5])`
      env, assert `observation([20, 0]) == [0.5, 0.5]`.
  (b) **Round-trip** — for random observations within the bounds,
      `denormalize(observation(x)) == x` to float32 tolerance.
  (c) **Zero-range guard** — when `low[i] == high[i]`, construction
      must raise `ValueError` (the code does this; lock the
      contract).
  (d) **Reset rebuild** — construct a `MockEnv` whose
      `observation_space` is reassigned between two `reset()`
      calls (simulating `ResampleBuildingOnResetWrapper`); assert
      the second `reset` re-reads the new bounds and the second
      `observation()` uses them.
- Acceptance: a sign flip or `(high + low)` typo in the affine map
  fails (a)–(b); a stale-bounds bug after `ResampleBuildingOnResetWrapper`
  swap fails (d).

#### T24b. `PadObservation`: metadata-driven zone split + reset rebuild

The existing tests use a `MockEnv` without `observation_names`, so
they always hit the *fallback* "last 7 features" heuristic. The
real code path — splitting on `obs_names[i].startswith("zone air
temperature")` — is never executed.

- Files: extend `tests/test_wrappers.py` (or new
  `tests/quick/test_pad_observation.py`) with:
  (a) **Metadata-driven split** — `MockEnv` with
      `metadata["observation_names"] = ["Zone Air Temperature Z1",
      "Zone Air Temperature Z2", "Outdoor Air Temperature", ...]`
      (using the real EnergyPlus naming). Assert
      `_zone_air_temperature_indices` returns `[0, 1]`, not the
      fallback range.
  (b) **Case / whitespace robustness** — same test with one name
      that has trailing whitespace and mixed case. Document the
      current behaviour (it lower-cases and strips) by locking it
      in a test.
  (c) **Non-zone features land at the end** — with the metadata
      from (a), assert `observation(obs)` puts the original
      non-zone values at indices `[max_zones :]` and pads zone
      slots `[current_num_zones : max_zones]` with zeros.
  (d) **Reset rebuild after resample** — `MockEnv` whose inner
      `observation_space` and `observation_names` both change
      between two resets (different zone counts); assert the
      padded output remains valid.
  (e) Rewrite or delete
      `test_pad_raises_error_if_too_many_zones` (currently doesn't
      exercise the zone-specific branch its name claims; see T4).
- Acceptance: a regression in `observation_names` capitalization,
  or in the zone/non-zone split, fails (a)–(c).

#### T24c.0. `AugmentObservationWithBuildingParams`: fail-loudly contract

Prerequisite for T24c (test). Resolves the silent-fallback
violation in `_extract_building_params` against `AGENTS.md`'s "no
silent fallbacks" rule.

Contract: **raise by default; opt-in fallback via a constructor
kwarg.**

```python
class AugmentObservationWithBuildingParams(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        building_params: dict[str, float] | None = None,
        *,
        allow_defaults: bool = False,
    ): ...
```

When `allow_defaults=False` (the new default), missing keys in the
extracted metadata raise `KeyError` with the missing key name.
When `True`, the current default-filling + warning behaviour is
preserved exactly.

Caller audit (mandatory in this commit) — covers **both production
and test call sites**:

- **Production call sites.** Grep every call site for
  `AugmentObservationWithBuildingParams` outside `tests/`.
  Anywhere multi-building training or eval currently relies on the
  silent fallback (likely
  `baselines/utils/training.py`, `baselines/eval_dynamics_adaptation.py`,
  and analysis scripts), pass `allow_defaults=True` *explicitly*.
  This preserves runtime behaviour while making the dependence on
  the fallback visible at the call site.
- **Test call sites.** Grep the same in `tests/`. Existing tests
  like
  `tests/test_wrappers.py::TestAugmentObservationWithBuildingParams::test_augment_with_custom_params`
  construct a `MockEnv` without full metadata; without
  `allow_defaults=True` they would break the moment T24c.0 lands.
  Patch every such test to set `allow_defaults=True` explicitly in
  this same commit. T24c (the test rewrite) replaces these
  patched calls with the new contract in a sibling commit, but
  T24c.0 must not be allowed to red-light the suite.
- Document the audit in the commit message: list every call site
  (production *and* test), whether it sets `allow_defaults=True`
  or relies on the new raise-by-default behaviour, and why.

- Files: `building2building/simulator/wrappers.py`; every caller
  identified by the audit.
- Acceptance: production behaviour unchanged for current callers
  (they all set `allow_defaults=True`); `pytest -m quick` green
  immediately after T24c.0 lands (no test starts raising); any
  *new* call site that forgets to provide full metadata raises
  immediately.

#### T24c. `AugmentObservationWithBuildingParams`: test the new contract

- Files: extend `tests/test_wrappers.py` (or new
  `tests/quick/test_augment_building_params.py`) with:
  (a) **Happy path** — env with full metadata; assert
      `building_params` matches and `normalized_params` are
      finite, in `[-1, 1]`.
  (b) **Missing metadata raises by default** — env with no `area`
      raises `KeyError` mentioning `"area"`.
  (c) **`allow_defaults=True` preserves old behaviour** — env
      with no `area`, constructed with `allow_defaults=True`,
      falls back to `100.0` and emits a `logger.warning`. (Use
      `caplog` to check the warning.)
  (d) **Reset re-extraction** — swap the inner env's `metadata`
      between two resets; assert `wrapped.building_params` and
      the observation space update on the second reset.
  (e) **Round-trip via `denormalize`** — assert
      `wrapper.denormalize(wrapper.observation(obs))` recovers the
      original obs (the wrapper strips the trailing param block
      and delegates to inner `denormalize` if present).
- Acceptance: silent-fallback contract is now explicit; the
  resample-swap path is exercised.

#### T24d. `ResampleBuildingOnResetWrapper`: zero → meaningful coverage

This wrapper is central to multi-building training (dynamics
adaptation, cross-domain transfer) and has *no tests*.

**`IndexError` contract (decided):** keep the current swallow-and-
resample behaviour, but make it audible. The fail-loudly principle
in `AGENTS.md` is real, but unilaterally removing the swallow
would crash multi-building training runs that today survive a bad
actuator on one of the buildings in the resample pool. The
contract for Phase T is:

- `step()` catches `IndexError`, **emits a `RuntimeWarning`**
  (via `warnings.warn`, *not* just `logger.warning` — that way it
  surfaces in CI test runs by default), returns
  `terminated=True, reward=0.0`, and resamples on the next
  `reset()`.
- The wrapper docstring is updated to document this branch as
  intentional, with a pointer to a follow-up phase (TZ2 or
  similar) for the deeper fix: emit a B2B-defined
  `ActuatorMismatchError` from the simulator dispatch site so the
  wrapper can catch a narrow exception class instead of bare
  `IndexError`.

T24d therefore covers both the production-code touchup and the
new tests in one commit.

- Files: `building2building/simulator/wrappers.py` (swap
  `logger.warning` for `warnings.warn(RuntimeWarning)` in the
  `step()` `except IndexError` branch; update docstring); new
  `tests/quick/test_resample_building_wrapper.py` using a tiny
  `FactoryMockEnv(index)` that records its index. Cover:
  (a) **Empty available_indices raises** —
      `available_indices=[]` raises `ValueError` at construction.
  (b) **Single-index is stable** — with
      `available_indices=[0]`, `reset()` does not swap the inner
      env (verify the factory is called exactly once across N
      resets).
  (c) **Multi-index sometimes swaps** — with
      `available_indices=[0, 1, 2]` and a seeded `random`,
      verify the factory is called again with a new index when
      `reset()` draws a different one; the previous env's
      `close()` is invoked.
  (d) **Episode counters reset** — after a 5-step episode and a
      `reset()`, `_episode_reward = 0.0` and `_episode_steps = 0`;
      `_episode_count` increments.
  (e) **`IndexError` is swallowed, warns, and resamples** — a
      factory that produces an env whose `step()` raises
      `IndexError` triggers a `RuntimeWarning` (assert with
      `pytest.warns(RuntimeWarning, match="actuator")`), returns
      `terminated=True, reward=0.0`, and the next `reset()` calls
      the factory again with a (possibly new) index.
  (f) **W&B no-op when inactive** — without an active `wandb` run,
      `reset()` and `step()` must not raise and must not import
      side-effects (use `monkeypatch.setitem(sys.modules, "wandb",
      None)` or similar).
- Acceptance: every code branch in the wrapper has at least one
  assertion; the `IndexError` swallow is documented as intentional
  in the docstring and surfaces a `RuntimeWarning` in CI; a
  follow-up item (TZ2) is filed for the `ActuatorMismatchError`
  refactor.

#### T24e. `wrap_env_for_rl`: composition + action round-trip

`wrap_env_for_rl` is the single source of truth for "wrap an env
for RL training". Today it is only tested against a real
`OfficeSmall` env (mis-marked as quick; see T3). After T0 lands we
can do this properly against the minimal fixture or a stub.

- Files: new `tests/quick/test_wrap_env_for_rl.py` using a `MockEnv`
  with `action_space=Box(low=[15.0, 5.0], high=[30.0, 25.0])` and
  `observation_space=Box(low=[10, -5], high=[30, 5])`:
  (a) **Composition order** — with `rescale_action=True,
      normalize_obs=True`, the *outermost* `observation_space` is
      `Box([0, 0], [1, 1])` and the *outermost* `action_space` is
      `Box([-1, -1], [1, 1])`. Document via assertion that
      `RescaleAction` is inner.
  (b) **Action round-trip** — pass `action=[-1, -1]`; capture the
      action the inner `MockEnv.step` receives (record it in a
      list); assert it equals `[15.0, 5.0]` (the engineering-unit
      low). Same for `action=[1, 1]` → `[30.0, 25.0]`. Same for
      `action=[0, 0]` → `[22.5, 15.0]` (mid-range).
  (c) **Independence of flags** — `normalize_obs=False` leaves the
      action stack unchanged; `rescale_action=False` leaves the
      observation stack unchanged.
  (d) **`metadata` passthrough** — set
      `env.metadata["observation_names"] = [...]` on the inner env;
      assert it is accessible on the wrapped env via `getattr`
      fallthrough.
- Acceptance: a wrong composition order, or a sign flip in
  `RescaleAction`, fails. Removes the need for the (currently
  mis-marked) `test_rl_wrappers.py` in `tests/quick/`.

### T25. Document wrapper expectations in `docs/guide/wrappers.md`

After T24a–T24e land, the wrappers' contracts are pinned by tests.
The user-facing wrapper guide should cite those contracts so
contributors know what is enforced (vs. what is convention).

- Files: `docs/guide/wrappers.md` (add a "Tested invariants"
  section per wrapper).
- Acceptance: `mkdocs build --strict` passes; each wrapper section
  in the guide cross-links to its test file.

### T27. Final sweep: audit, delete, relocate

Merged T26 + T27. The principles in *Test design principles for
Phase T* were codified after T1–T25 were drafted, and Phase T adds
~20 new tests and rewrites several existing ones — so a single
deliberate pass at the end of the phase catches everything that
slipped through.

T27 ships as two commits:

- **T27a: audit + checklist artifact.** Walk the checklist below
  and produce `tests/PHASE_T_SWEEP.md` — a checked-in file
  listing, per item, the exact files / tests / fixtures to
  touch and the decision taken. T27a does not change any test
  file; it only commits the checklist. A reviewer can sign off
  on the audit independently of the execution.
- **T27b: execute the checklist.** Apply every decision recorded
  in T27a's checklist. The PR diff should match the checklist
  one-to-one. If during execution a decision turns out to be
  wrong, update the checklist in the same commit and explain in
  the commit message.

This structure prevents the "audit produces a list, sweep happens
later" failure mode (split PRs) *and* prevents the inverse "sweep
without an audit artifact" failure mode (reviewer has to
reverse-engineer the decision tree from the diff).

Concrete sweep checklist (T27a populates this with file-level
detail; T27b executes):

1. **No tests directly under `tests/`** other than `conftest.py`.
   The two legacy files left at the top level after T2 —
   `tests/test_get_actuators.py` and `tests/test_wrappers.py` (T2
   already absorbed `test_building_param_clipping.py`) — must be
   either deleted (if T7/T9/T24 already cover what they exercise)
   or moved into `tests/quick/` or `tests/long/` with appropriate
   markers. After this step `ls tests/*.py` returns only
   `conftest.py`.
2. **Delete `tests/long/test_pipeline_single_zone_houses.py`.**
   Under the new tier definition it is not `long` (max_episode_steps
   = 8, single `step()` call); under any definition it asserts only
   that `reset()` and `step()` don't crash, which T5's SFH
   parametrize case + T20's benchmark smoke test cover with
   actual pipeline assertions. Mention the deletion in the commit
   message so the SFH-specific path is greppable.
3. **`tests/quick/test_eval_bugs.py`** — re-evaluate the whole
   file. `TestParseModelPath::test_rglob_discovers_nested_models`
   and `test_standard_nested_structure` encode a real path-layout
   contract worth keeping. Everything else (the CSV-column rename,
   the `pad_obs_size` keyword-only check, the `metadata.json`
   roundtrip) tests historical bug fixes whose absence won't break
   anything today — delete those and rename the surviving slice to
   `tests/quick/test_eval_path_layout.py`. If nothing survives,
   delete the file.
4. **Convert any remaining `MagicMock(spec=<B2B class>)` to real
   constructors.** Grep `tests/` for `MagicMock(spec=` and
   `MagicMock(spec_set=`; for any that target a B2B class
   (`BuildingInfo`, `TaskConfig`, `EnvBuildConfig`,
   `BuildingConfig`, etc.), replace with a real instance built via
   `from_dict` / a constructor + the T0 fixtures. Mocks of generic
   `gym.Env` are fine (Principle 5).
5. **Convert any hand-rolled config dict** to its real
   `from_dict` / constructor path. Grep for dict literals shaped
   like `{"target_temperature_mode": ..., "reward_config": ...}`
   in test files; if the test means to construct a
   `TaskConfig` / `EnvBuildConfig`, do so directly.
6. **Wrapper companion-test audit.** For T24a/b/c/e (the wrapper
   tests that ship using `MockEnv`), add one companion test per
   wrapper that runs the same wrapper against a real env built
   from `fixture_registry`. Each companion test is short (~10
   lines): build env, wrap, reset, assert one invariant. The goal
   is to catch wrappers that work on `MockEnv` but break on the
   real `metadata` / `observation_space` shape. If T24's `MockEnv`
   tests already use the real metadata structure (T24b explicitly
   does), the companion test there is a one-liner asserting that
   the metadata field shape matches the production env's; don't
   duplicate.
7. **Stale fixture audit.** With the T0 minimal fixture and the
   T9 per-HVAC-type equipment fixtures committed, several existing
   fixture files are likely unused: `tests/fixtures/bldg1.epjson`,
   `tests/fixtures/bldg1-setpoint-control/`,
   `tests/fixtures/eplusout.edd`, `tests/fixtures/eplustbl.htm`,
   `tests/fixtures/in.schedules.csv`, `tests/fixtures/weather.epw`,
   `tests/fixtures/bldg2.epjson`.
   For each, run `rg <basename> tests/` — if no remaining test
   references it, `git rm` it. Document the deletions in the
   commit message so a contributor who later needs an EDD fixture
   knows it used to exist.
8. **Re-evaluate `test_reward_normalizers.py::test_default_path_constant_points_inside_package`.**
   T1 kept it; T27 decides if it has earned its place now that
   `data/` should have stabilised. Keep if `data/` is still moving;
   delete if the layout has been frozen by some other phase.
9. **No test takes longer than its tier promises.** Time
    `pytest -m quick --durations=20`. Targets:
    - Total `pytest -m quick` wall-clock: ≤ 90 s on the dev box
      (aim for ~60 s, allow buffer for CI variance).
    - Any single `quick` test > 10 s gets reviewed: either split,
      or move to `long/` with justification, or document why the
      10 s is unavoidable in the test docstring.
    Robustness over speed — these are guidelines, but every
    violation must be a deliberate, documented choice, not an
    accident.
10. **Every remaining test file has a module docstring** explaining
    *what contract it pins*, not what code it imports. A test whose
    docstring is "tests for `xyz` module" is a smell — rewrite it
    to "asserts that <invariant>".
11. **No `pytest.mark.skip` / `pytest.mark.xfail` without a
    tracked TODO**. If a test must be skipped, the skip reason
    must reference an open TODO item (or be deleted along with
    the code it tested).

- Files (T27a): `tests/PHASE_T_SWEEP.md` (new).
- Files (T27b): every test file in `tests/` per the checklist;
  deletions / relocations / companion tests / docstring rewrites.
- Acceptance (T27a): the checklist is committed, every checklist
  item is a concrete decision (file path + action + rationale),
  no test file is modified.
- Acceptance (T27b): `ls tests/*.py` returns only `conftest.py`;
  `rg "MagicMock\(spec=" tests/` returns no B2B classes;
  `rg "pytest.mark.skip|pytest.mark.xfail" tests/` returns only
  entries with linked TODOs; every surviving test file has a
  contract-focused module docstring; the Phase T summary in the
  T27b commit message lists net test count and total
  `pytest -m quick` wall-clock before vs. after the phase.

### T28. Document the new tests in `docs/about/testing.md`

After T1–T27 land, update the testing page to reflect the new
inventory and tier definitions:

1. **Rewrite the tier definitions table** to use the new
   rollout-length definition (see "Tier definitions for Phase T"
   at the top of this section). Make explicit that `quick`
   includes EnergyPlus.
2. **Add a `release` tier section** linking to `tests/release/`
   and pointing at deferred item TZ1 for the CI automation.
3. **Remove deleted-test entries** and add new-test entries.
4. **Update the "intentionally not covered" section** —
   pipeline / wrappers / etc. have moved out of that list during
   Phase T.
5. **Surface the *Test design principles for Phase T* block** on
   the testing page so future contributors see it without
   having to read `TODO.md`. Reproduce the block verbatim under
   a "Test design principles" heading in `docs/about/testing.md`
   — mkdocs has no first-class transclude, and a stale snapshot
   is preferable to a broken include directive. Add a note at
   the top of the section pointing at `TODO.md` Phase T as the
   canonical source for as long as Phase T is in flight, and
   remove the pointer once the phase closes.

- Files: `docs/about/testing.md`.
- Acceptance: `mkdocs build --strict` passes; the gap analysis at
  the bottom of the page shrinks visibly; the principles and the
  three-tier definition are reproduced (or linked) on the testing
  page.

### T29. (Dropped.)

A grep-based public-function coverage script was considered
(import each `__all__` symbol, look for it in `tests/`, fail on
zero hits) but dropped from Phase T. The grep approach has
well-known false negatives — a symbol exercised transitively
through `new_make_env` does not appear by name in any test — and
in practice every false positive ends up on an exemption list
that grows until the script is a rubber stamp.

If a coverage gate becomes worthwhile after the suite has
stabilised, the right tool is `coverage.py` with a per-file line-
coverage threshold (e.g. `api/__init__.py ≥ 80%`,
`pipeline/*.py ≥ 60%`), not a grep script. That decision is
deferred until after Phase T closes and the testing.md page
reflects the new inventory.

---

## Landing the `feature/reward-normalization` branch

The branch is 5 commits ahead of `origin/dev`, 0 behind, not yet
pushed. It mixes the reward redesign, an RL wrapper refactor, eval
bug fixes, and a SAC baseline. Return it to `dev` as a stack of
small reviewable PRs cut from `dev` (not stacked in git; each is
logically independent). Use the `split-to-prs` skill when slicing
the diff. After PR-1..PR-3 land, delete the branch; subsequent
Phase B/C/D work continues on short-lived branches off `dev`.

### L0a. Docs housekeeping commit on `dev`

Lands directly on `dev` to unblock everything else.

- Files: `AGENTS.md`, `design_doc.md`, `TODO.md`, `notes.md`,
  `paper/reviewer_feedback.md`, the two `.gitignore` fixes
  (`wandb/` → `/wandb/`, `analysis/` → `/analysis/`), and the
  pre-existing `.cursorrules` / `.augment/` deletions.
- Acceptance: pushed to `dev`; no behaviour change.

### L0b. Promoted-ignored commit on `dev`

- Files: `baselines/configs/wandb/default.yaml` (required by Hydra
  defaults; was silently ignored). Decision needed on
  `baselines/analysis/`: delete or track.
- Acceptance: Hydra resolves `defaults: - wandb: default`; decision
  on `baselines/analysis/` documented in `notes.md`.

### L1. PR-1 — Normalized deadband reward

Content of `799aeb1 reward normalization w.r.t. reactive control`,
plus the YAML-rename portion of `ad81ad0 wip`
(`reward_normalizers.yaml` → `reward_normalizers_linear.yaml`,
add `reward_normalizers_random_linear.yaml`).

- Files: `building2building/{types,config/tasks,simulator/__init__,
  simulator/rewards,data/reward_normalizers,api/__init__,
  benchmarks/goal_adaptation}.py` and the four reward tests.
- Acceptance: branch cut from `dev`; `pytest -m quick` green;
  lands the new reward family **alongside** (not in place of) the
  legacy family — D2 happens later in Phase D-house.

### L2. PR-2 — RL wrapper stack + eval bug fixes

`make_rl_env_fn`, `api/rl_wrappers.py`, `wrap_env_for_rl`, the
bug-fix slice of `eval_ppo.py` / `eval_dynamics_adaptation.py` /
`tune_ppo.py`, plus `tests/quick/test_rl_wrappers.py`.

- Files: as above.
- Acceptance: branch cut from `dev`; `pytest -m quick` green; does
  not depend on Phase A; unblocks Phases B and C.

### L3. PR-3 — SAC baseline (current state)

`baselines/train_sac.py`, `configs/experiment/train_sac*.yaml`,
`configs/training/sac.yaml`, `configs/policy/sac.yaml` (current
state, with critic-stability fixes),
`tests/quick/test_train_sac_smoke.py`.

- Files: as above.
- Acceptance: branch cut from `dev`; `pytest -m quick` green;
  README flags SAC as "unstable on occupancy-driven tasks; final
  tuning in PR-4".

### L4. PR-4 — SAC config under final normalized reward (deferred)

Lands after Phase A (reward locked) and Phase B (SAC ablation B2
on the calibration anchor completes). The only PR in the stack
whose contents are not already on the branch.

- Files: `baselines/configs/policy/sac.yaml` (final values),
  README "Known Issues" cleanup.
- Acceptance: B2 result documented in `notes.md`; PR cut from `dev`.

---

## Deferred — pick up in a later phase

### TZ1. CI wiring for `tests/release/`

Phase T introduces a third pytest tier (`release`) for data-integrity
checks (T15, T16) that require the published HuggingFace dataset and
should not run on every push. The marker, the directory, and the tests
themselves land in Phase T (T15.0–T16), but the *automation* — a
GitHub Action (or equivalent) that runs `pytest -m release` on
release-candidate tags, plus a one-paragraph entry in `REPRODUCING.md`
explaining the manual invocation — is deferred so Phase T stays
focused on the test suite itself.

- Files: `.github/workflows/release-checks.yml` (new),
  `REPRODUCING.md` (add a "Release validation" section pointing at
  the workflow and the manual `pytest -m release` command).
- Acceptance: pushing a tag matching `v*` triggers the workflow;
  failures block the release; `REPRODUCING.md` documents both the
  automated and manual paths.

### TZ2. Emit `ActuatorMismatchError` from the simulator dispatch site

Phase T's T24d keeps the existing `IndexError` swallow in
`ResampleBuildingOnResetWrapper.step` (with a `RuntimeWarning`) as
an explicit, documented backwards-compat decision. The deeper fix
is to stop catching a bare `IndexError` — which can mask unrelated
bugs in the inner env — and instead catch a B2B-defined exception
class raised from the precise call site where the actuator-set
mismatch is detectable.

- Files: define `ActuatorMismatchError` in
  `building2building/simulator/` (next to the dispatch site);
  raise it from the dispatch path that currently lets `IndexError`
  propagate; narrow the wrapper's `except` clause from
  `IndexError` to `ActuatorMismatchError`; update T24d's test (e)
  to assert on the narrower exception class.
- Acceptance: a non-actuator `IndexError` raised by the inner env
  now propagates instead of being swallowed; T24d's
  `pytest.warns(RuntimeWarning, match="actuator")` test still
  passes; the `RuntimeWarning` message names the building index
  and the mismatched actuator.

### TZ3. (Conditional) Parallelize the multi-zone leak test

Filed **only if** T22a's measurement on `OfficeMedium` shows that
even `_N = 3` exceeds the 5-minute budget. T22b in that case
ships a parallelization refactor (subprocess fan-out via
`multiprocessing`, or a `pytest-xdist`-style worker pool) and
defers the actual OfficeMedium parametrize to TZ3.

- Files: `tests/long/test_env_leak.py` (parametrize
  `_BUILDING_TYPE` over `SingleFamilyHouse` and `OfficeMedium`
  using the new parallel harness from T22b).
- Acceptance: OfficeMedium leak coverage is in the long suite
  with `_N ≥ 5` and total runtime under 5 minutes on the
  parallel harness; the per-cycle wall-clock measurement is
  documented in the test's module docstring and the commit
  message.
- Do not file this item until T22a's measurement is recorded. If
  T22b's measurement permits `_N ≥ 5` without parallelization,
  TZ3 is moot and should not be filed.

### TZ4. Coverage gate via `coverage.py` line-coverage thresholds

Phase T's T29 was dropped in favour of deferring the coverage-
gate question until after the suite has stabilised. Once Phase T
closes and `docs/about/testing.md` reflects the new inventory
(T28), revisit whether a coverage gate is worthwhile, and if so
implement it with `coverage.py` per-file thresholds — not a
grep-based script.

The right framing is "lines covered" per file, with thresholds
calibrated against the post-Phase-T baseline (so the gate
ratchets up over time, not down). A reasonable starting matrix:

- `building2building/api/__init__.py` ≥ 80 %
- `building2building/pipeline/*.py` ≥ 60 %
- `building2building/simulator/wrappers.py` ≥ 90 % (Phase T
  brings this close to 100 % via T24a–T24e)
- Everything else: no hard threshold, but a `coverage report`
  artifact uploaded on every CI run for trend visibility.

- Files: `pyproject.toml` (configure `[tool.coverage]` with
  `source`, `omit`, and per-file thresholds), CI workflow
  (run `coverage run -m pytest -m quick && coverage report
  --fail-under=...`), `docs/about/testing.md` (one paragraph
  pointing at the gate and listing the thresholds).
- Acceptance: a deliberate coverage-reducing change in any
  file with a threshold fails CI; the trend artifact is
  visible on every CI run; the gate is documented.
- Do not file before Phase T closes — calibrating thresholds
  against a moving suite produces an unstable gate.
