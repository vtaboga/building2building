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

Choice: `reward_normalizers_random_linear.yaml` (SAC-warmup uniform
random). Rationale: policy-independent; bakes in no RBC-specific bias.
Tuned-RBC YAML archived to `building2building/data/archive/`.

### ~~A2. Re-run calibration if A1 invalidates the current YAML~~ N/A

A1's choice matches the pre-existing default; no re-run needed.

### A3. Validate and commit the calibration sanity plot

- Files: regenerate `analysis/task_study/reward_design/plots/fig_normalizer_calibration.png`
  from the chosen YAML; commit it (or document its location if it
  lives outside the release tree).
- Acceptance: per-bucket median of `temp_penalty/τ_T` and
  `power_penalty/τ_E` is exactly 1.0 by construction; bucket IQRs
  match the values in the committed YAML.

---

## Phase B — Stabilize RL training under the new reward

### B0. Diagnose and fix the EnergyPlus resource leak (gates B2/B3/B4 and all of Phase C)

Long-running processes that create many envs sequentially (Optuna
tuning, SAC ablation, PPO sweeps, `run_reactive_control.py` over the
full catalog) currently hit two distinct leaks:

1. **Filesystem leak.** Each new `EnergyPlusEnvironment` writes
   `eplusout.*` artefacts under its `eplus_output_dir` (typically
   `$SLURM_TMPDIR` / `$SCRATCH`) and never removes them, so SLURM jobs
   eventually fail with "no space left on device" or hit per-user
   inode/file-count quotas on tmpfs. The directory is passed to
   `MakeEnergyPlus` at construction time and is not stored on the env,
   so even an upstream `close()` fix in `minergym` cannot reach it.
2. **Thread + native-state leak.** `EnergyPlusEnvironment.close` is the
   inherited `gym.Env` no-op: it neither calls `self.ep.try_stop()` nor
   joins the daemon `threading.Thread` running
   `api.runtime.run_energyplus`. The thread keeps alive its closure
   (and the captured `ManagedState`), which holds the native EnergyPlus
   state, so RAM grows linearly with the number of envs created.

Both modes are currently *partially patched* by
`baselines/utils/evaluation.py::close_env_aggressively`, which drills
through the wrapper stack, calls `try_stop`, joins the thread,
`gc.collect()`s, and `shutil.rmtree`s the output dir. This wrapper is
load-bearing for `train_sac.py`, `tune_controller.py`,
`run_reactive_control.py`, `tune_ppo.py`, and the analysis scripts —
but it is opt-in: any new entrypoint that forgets to call it leaks
silently, and the per-call thread-join timeout sometimes triggers the
"thread did not exit" warning under load.

This task is to (a) reproduce both leaks under a controlled harness,
(b) push the fixes upstream where possible, (c) make the in-tree
behaviour leak-free by default rather than opt-in.

- Files (investigation):
  - `baselines/utils/evaluation.py::close_env_aggressively`
    (existing diagnosis lives in its docstring — start there).
  - `minergym` source (vendored or pip-installed):
    `minergym/environment.py::EnergyPlusEnvironment.{close,reset}`,
    `minergym/runtime.py` (`MakeEnergyPlus`, `run_energyplus`,
    `ManagedState`, `eplus_output_dir` plumbing).
  - `building2building/api/__init__.py::new_make_env` and
    `building2building/envs/factory.py` (where `eplus_output_dir`
    originates and could be tracked on the returned env).
  - `building2building/simulator/wrappers.py::ResampleBuildingOnResetWrapper`
    (calls `self.env.close()` on every building swap — currently
    relies on the upstream no-op and therefore leaks on every reset
    that draws a new index).
- Files (likely fix surface, scope to be confirmed during
  investigation; **do not pre-commit to a shape**):
  - Upstream patch in `minergym`: override `EnergyPlusEnvironment.close`
    to `try_stop`, join the thread, `self.ep = None`. Submit as a PR
    upstream; pin the new version in `pyproject.toml`.
  - In-tree: have `new_make_env` / the factory remember the
    `eplus_output_dir` on the returned env (e.g. attribute
    `_b2b_eplus_output_dir`) so `close()` can also `rmtree` it without
    the caller knowing the path.
  - Make `ResampleBuildingOnResetWrapper.reset()` call the same
    full-cleanup path so long training runs do not leak per-episode.
  - Once the default `close()` is leak-free, downgrade
    `close_env_aggressively` to a thin compatibility shim with a
    `DeprecationWarning`; keep it for the camera-ready PRs but mark
    it for removal in D-phase.
- Acceptance:
  - A new test under `tests/long/` (gated, since it needs EnergyPlus)
    creates and closes N=20 envs in a loop and asserts:
    (i) the parent `eplus_output_dir` contains 0 leftover subdirs after
    each close;
    (ii) `threading.active_count()` returns to its baseline within
    `thread_join_timeout`;
    (iii) RSS growth across the loop is bounded (e.g. <50 MB total via
    `psutil.Process().memory_info().rss`).
  - The same test passes with **plain `env.close()`** (no
    `close_env_aggressively`) once the fix lands.
  - A 6-building × 3-seed SAC run completes without the
    "EnergyPlus thread did not exit within Xs" warning firing.
  - The B2 ablation matrix runs to completion on a single SLURM node
    without exhausting `$SLURM_TMPDIR`.
- References: `baselines/utils/evaluation.py` lines 114–205
  (existing diagnosis); `notes.md` § "Operational gotchas".

### B1. Apply SAC fixes 1–3 to the policy config

- Files: `baselines/configs/policy/sac.yaml` —
  `log_std_init: 0.0`, `ent_coef: 0.2`, `use_sde: false`. Add an
  inline comment block citing the diagnostic findings.
- Acceptance: `pytest -m quick tests/quick/test_train_sac_smoke.py`
  passes; SAC training launcher still produces valid trajectories
  on a 50 k-step smoke test.
- Reference: `notes.md` § "SAC".

### B2. SAC ablation on the calibration anchor

Run the 5-cell × 3-seed × 6-building ablation matrix on
`task_occ_wmed`, full-year, to confirm the fixes restore stable
learning. Commit a result note.

- Files: `analysis/task_study/sac_diagnostic/` (data + plots),
  update `notes.md` with the chosen final SAC config.
- Acceptance: 0/18 (or near-zero) runs show
  `best - final > 0.1`; action saturation stays below 10% at every
  checkpoint; plot in `notes.md` quick-reference table.

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

### D4. Drop `baselines/requirements.txt`; update install docs

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
