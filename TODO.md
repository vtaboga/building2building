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

### D3. Fix the documented eval bugs

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

### D5. Write `REPRODUCING.md`

A single document mapping every paper figure / table to the exact
command (Hydra invocation, expected outputs, plotting command).

- Files: new `REPRODUCING.md` at repo root; cross-link from
  `README.md` and `design_doc.md`.
- Acceptance: each Phase-C deliverable (PPO specialist, dynamics
  adaptation, cross-domain, baseline_returns.csv) has an entry
  with the exact one- or two-line command.

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
