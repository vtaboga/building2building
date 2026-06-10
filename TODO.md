<!-- -*- mode: markdown -*- -->

# TODO

Remaining work toward the open-source release and paper rerun under the
normalized reward. **One TODO, one commit** (per `AGENTS.md`).

Two phases remain. **Phase B (reward-coefficient study) must complete
before Phase A (paper rerun)**: the final `emed` / `ehigh` values must be
pinned in `building2building/config/tasks.py` so every Phase-A artefact is
produced by the locked coefficients. Every Phase-A artefact must be
reproducible by the public OSS code, not by the in-flight branch.

Phases M (OfficeMedium OA-mixer fix), D-house, T (test suite), F (file/doc
audit), and D-API (public surface) are complete — their history lives in
git.

The LaTeX paper itself lives outside this repository (`paper/` was removed
in `3a808c5`). Phase A / B produce the figures and CSVs the paper consumes;
the paper edits themselves are out of scope for this repo.

Format per item: short title; affected files; acceptance check; references.

---

## Phase A — Re-run paper experiments (FINAL stage, on the OSS codebase)

**Prerequisite — must be merged before A1 starts:** Phase B, so the final
`emed` / `ehigh` values are pinned in `building2building/config/tasks.py`
and every Phase-A run uses the locked coefficients. (Phases M, D-house, T,
and F are already complete.)

The OfficeMedium fix (Phase M) is in, so OfficeMedium rows are produced
against the correct action space.

### A1. Regenerate `baseline_returns.csv` on the **full** split

Run `baselines/run_reactive_control.py` across **every**
`(building_type, building_id, task, run_period)` tuple in the
full train + test split, under the locked `emed` / `ehigh`
values from Phase B. The reactive baseline is the reference
controller for the normalized score and underlies every figure
in the paper — both the specialists (A2) and the transfer
benchmarks (A3 / A4), so train-split rows are not optional.

- Files (release artefact): `building2building/scores/
  baseline_returns.csv` (replaced in full); schema unchanged from
  the existing header.
- Files (release entry point): `baselines/run_reactive_control.py`
  + the existing
  `baselines/configs/experiment/eval_reactive_control.yaml`. The
  full reproduction command (in `REPRODUCING.md` § A1) is the
  single-machine Hydra invocation:

      python -m baselines.run_reactive_control \
          experiment=eval_reactive_control \
          output_csv=building2building/scores/baseline_returns.csv

- Files (developer convenience): `baselines/scripts/
  run_baseline_returns.sh` Slurm wrapper for the cluster (a thin
  loop around the Python entry point — not the canonical
  reproduction step). Update the Slurm script's header to iterate
  the full 9-cell task grid + 3 run periods + the full building
  catalogue; user submits.
- Tasks × run periods: full 9-cell grid
  (`task_{const,occ,rand}_{e0,emed,ehigh}`) × `run_period ∈
  {full_year, winter, summer}`. Total row count ≈
  `len(full_split) × 9 × 3`.
- Acceptance: every `(building_type, building_id, task,
  run_period)` tuple the paper figures cite is present;
  `pytest -m quick tests/quick/test_scoring.py` passes;
  `tests/release/test_baseline_returns_coverage.py` passes
  against the new CSV; `REPRODUCING.md` § A1 is updated to point
  at the locked-`emed`/`ehigh` values.

### A2. PPO + SAC specialists on `test_small` (paper §5)

Scope: **1 seed, default params, no tuning, full 9-task grid,
evaluated on the test_small subset of every building type.**

Total cells = `len(building_types) × len(test_small per type) ×
9 tasks × 2 algorithms × 1 seed`. With 6 building types and ~5
test_small buildings per type, that is `6 × 5 × 9 × 2 = 540`
training runs. Each PPO specialist run is ~5M steps × 14 envs;
each SAC specialist run is ~1M steps × 4 envs. CPU-only.

The reduction from the camera-ready original plan (3 seeds, full
test split) to (1 seed, test_small) is deliberate compute
shedding — the figure caption must state the seed count and the
building subset.

- Files (release artefacts): `baselines/train_ppo.py`,
  `baselines/train_sac.py` Hydra outputs collected under
  `outputs/`; merged CSVs `results_ppo_specialist.csv` +
  `results_sac_specialist.csv`; combined PPO+SAC specialist
  figure rendered by an extended
  `baselines/plotting/plot_ppo_specialist.py` (or a new
  `baselines/plotting/plot_specialists.py` covering both
  algorithms — pick one in the A2 first commit).
- Files (release entry points, cited from `REPRODUCING.md`):

      python -m baselines.train_ppo experiment=train_ppo_task_study \
          --multirun seed=0 \
          "tasks=[task_const_e0,task_const_emed,task_const_ehigh,task_occ_e0,task_occ_emed,task_occ_ehigh,task_rand_e0,task_rand_emed,task_rand_ehigh]" \
          building_split=test_small

      python -m baselines.train_sac experiment=train_sac_task_study \
          --multirun seed=0 \
          "tasks=[...]" \
          building_split=test_small

  Both Hydra entry points must support the `building_split=`
  override (today they may hard-code "train"); land that as the
  first A2 commit if missing.
- Files (developer convenience): existing
  `baselines/scripts/train_ppo_small_test_array.sh` and
  `baselines/scripts/train_sac_array.sh` are updated in-place
  (no `_v2` forks per `AGENTS.md`) to wrap the Python entry
  points above for cluster submission. They do **not** ship as
  the canonical reproduction step.
- Acceptance: the specialist figure regenerates from a single
  Python command; the figure shows PPO and SAC on the same axis;
  numerical values for the paper table can be cited from the
  merged CSVs; seed count (1) and `test_small` restriction are
  explicit in the figure caption; `REPRODUCING.md` § A2 + § A2-bis
  are updated to reflect the locked `emed`/`ehigh` values and the
  combined-figure command.

### A3. Dynamics adaptation (paper §6.1) — final step

Runs as the **final-step batch** after A1 and A2 are complete,
on exactly the same OSS-ready codebase.

Re-run the three approaches (specialist, baseline, parameterized)
at three difficulty levels (easy = SingleFamilyHouse, medium =
OfficeSmall, hard = OfficeMedium) under the new reward and the
new OfficeMedium action space. 1 seed, default params, matching
the A2 compute-shedding rationale.

- Files (release entry points, cited from `REPRODUCING.md` § A3):

      python -m baselines.train_dynamics_adaptation \
          experiment=train_dynamics_specialist difficulty=easy
      python -m baselines.train_dynamics_adaptation \
          experiment=train_dynamics_baseline difficulty=easy
      python -m baselines.train_dynamics_adaptation \
          experiment=train_dynamics_parameterized difficulty=easy

  (Repeat with `difficulty=medium,hard`.) Plotting via
  `baselines.plotting.plot_dynamics_adaptation` — see
  `REPRODUCING.md` § A3 for the full invocation.
- Acceptance: `fig_transfer_rew` and `fig_transfer_temp_deviation`
  regenerate from a single Python command per
  `REPRODUCING.md`; seed count and OfficeMedium action-space
  update are noted in the figure caption.

### A4. Cross-domain Amorpheus (paper §6.2) — final step

Runs alongside A3 as part of the final-step batch.

- Files (release entry points, cited from `REPRODUCING.md` § A4):

      python -m baselines.train_cross_domain experiment=train_cross_domain

  followed by the eval and plotting commands documented in
  `REPRODUCING.md` § A4.
- Acceptance: the cross-domain figure regenerates from a single
  Python command per `REPRODUCING.md`; the action-space update is
  noted in the figure caption.

---

## Phase B — Empirical reward-coefficient study

Output: final values of `emed` and `ehigh` (and confirmation that
`e0 = 0.0`), backed by empirical evidence on a policy ladder and
demonstrated invariance across the `test_small` building subset.

The user-confirmed methodology: reuse
`analysis/task_study/reward_design_comparison_sac.py` to train a
SAC agent while monitoring the per-term reward distribution over
training. The SAC trajectory itself provides the random →
optimal policy ladder (early training is near-random, late
training is the SAC optimum for that building).

Because B3's outputs land in the paper appendix and `baselines/`
is the release surface (`analysis/` is dev-only scratch), B1/B3
also execute the relevant slice of the deferred `analysis/` →
`baselines/` migration: the reward-study training script and its
plotters move into `baselines/`, rewritten against the public
`building2building` API. Any other paper-cited `analysis/` plotter
discovered while doing this moves alongside them.

### B0. Pin the empirical question

**Before any rollout.** Write a short methodology note
(`notes.md` § "Phase B methodology") that answers:

1. **Target invariance criterion.** What quantitative claim
   does B make? Recommendation:
   - For `emed`: at the SAC final checkpoint, the ratio
     `mean(temp_penalty / tau_T) / mean(emed * power_penalty /
     tau_E)` is in `[0.5, 2.0]` on every `test_small` building,
     i.e. the two reward terms are within a factor of 2 of each
     other end-to-end.
   - For `ehigh`: at the SAC final checkpoint, the same ratio
     drops below `0.5` (energy term dominates) on every
     `test_small` building.
   - For `e0`: trivially satisfied (`power_penalty` term has
     zero weight by construction); B0 only verifies that the
     temperature term behaves sensibly.
2. **Building subset.** `test_small` per building type. For
   compute reasons, B can restrict to a sub-subset (e.g. 2
   buildings per type, 12 total) for the *value search* and
   then validate on the full `test_small` for the chosen
   values. Pin the split in B0.
3. **Run period.** `full_year` (matches the reward-normalizer
   calibration regime). `winter` is faster but the calibration
   regime is `full_year`, so use `full_year`.
4. **SAC config.** Default from `baselines/configs/policy/sac.yaml`
   + `baselines/configs/training/sac.yaml`. No tuning.
5. **Coefficient grid to test.** Start with the current
   `{emed=1.0, ehigh=5.0}` plus a small grid around each:
   - `emed ∈ {0.5, 1.0, 2.0}`
   - `ehigh ∈ {3.0, 5.0, 10.0}`
   Total cells: `3 × len(building_subset)` for each coefficient
   (since the SAC trajectory itself spans the policy ladder, only
   one run per cell is needed).

- Files: `notes.md` § "Phase B methodology".
- Acceptance: every numbered choice above is pinned with a
  concrete value before any Slurm submission.

### B1. Productionize the SAC reward-distribution script into `baselines/`

`analysis/task_study/reward_design_comparison_sac.py` already
trains a SAC agent and decomposes the per-step reward into
`(temp_penalty, power_penalty)`. Since the **outputs of B3 land
in the paper appendix**, the final script must live under
`baselines/`, not `analysis/`. B1 is the productionization commit.

Concretely:

1. Promote the script to **`baselines/run_reward_coefficient_study.py`**
   (or a similar name; pin in B1's first commit). Move only the
   logic that produces the appendix data; leave any prototype-
   only branches behind in `analysis/` (they remain there for the
   audit trail).
2. Parametrize via Hydra (matching the rest of `baselines/`):
   the energy-weight grid, the building subset, the run period,
   and the SAC config. The Hydra entry point becomes the
   canonical Python invocation cited in `REPRODUCING.md`.
3. Extend the building loop to iterate the B0-pinned subset (a
   sub-subset of `test_small`).

- Files: new
  `baselines/run_reward_coefficient_study.py`; new Hydra config
  `baselines/configs/experiment/reward_coefficient_study.yaml`;
  the prototype under
  `analysis/task_study/reward_design_comparison_sac.py` is
  preserved as historical artefact. Update `REPRODUCING.md`
  in the same commit (new "Appendix: reward-coefficient study"
  section, with the Python invocation).
- Acceptance: a dry-run with one building × one energy weight
  produces the expected NPZ artefact under
  `outputs/reward_coefficient_study/`; the file size, schema,
  and column names match the prototype's output (so B3's
  plotting code can read it).

### B2. Run the sweep

The user runs the sweep from the Python entry point committed in
B1. The canonical invocation in `REPRODUCING.md` is the
single-machine command:

    python -m baselines.run_reward_coefficient_study \
        experiment=reward_coefficient_study

Expected runtime on the cluster:
`~12 buildings × 3 emed values × 3 ehigh values × ~1 hour SAC =
~108 hours of CPU time`; with 12 parallel cluster workers, ~9
hours wall-clock. A thin Slurm wrapper under
`baselines/scripts/run_reward_coefficient_study.sh` (developer
convenience only) loops the Python entry point per cell.

- Acceptance: every `(building, emed, ehigh)` cell produces a
  valid NPZ under `outputs/reward_coefficient_study/`; total
  compute documented in `notes.md`.

### B3. Aggregate + plot

The plotting code is part of the OSS release (it produces the
paper appendix figures), so it lands under `baselines/plotting/`,
not `analysis/`.

Produce:

1. **Ladder plot.** Per building, per energy_weight: `(temp_penalty / tau_T)`
   and `(w_E * power_penalty / tau_E)` over training, both on
   the same y-axis. Reader sees both terms move from "random"
   (early SAC) to "optimal" (late SAC).
2. **Balance plot.** Per energy_weight (averaged across the
   building subset): the ratio of the two terms at the final
   SAC checkpoint, with a band over buildings for invariance.
3. **Per-building invariance plot.** At the chosen final `emed`
   and `ehigh`: the ratio per building, error bars over
   training-final variance, on one axis. Reader sees that the
   ratio sits in the B0-pinned window on every building.

- Files: new
  `baselines/plotting/plot_reward_coefficient_study.py`;
  appendix-quality figures saved to the plotting output directory.
  Update `REPRODUCING.md` with the plotting command alongside
  B1's training command.
- Acceptance: the three figures regenerate from one Python
  command; the values pinned in B0 are visibly satisfied (or, if
  not, B3 recommends a different pair); the figures are at
  paper-appendix quality (font size, axis labels, units).

### B4. Pin final values + update presets

Based on B3, commit the final `emed` and `ehigh` values to
`building2building/config/tasks.py` (in `TASK_PRESETS`'s
`NormalizedDeadbandRewardConfig.energy_weight` fields for the
`task_{const,occ,rand}_{emed,ehigh}` presets). Update the
README task-preset table and the docstrings that cite these
values.

- Files: `building2building/config/tasks.py`, `README.md` (task
  table values), `docs/api/types.md` (if it cites the values).
- Acceptance: the values in the code match the B3 conclusion;
  `pytest -m quick tests/quick/test_task_presets.py` passes;
  `REPRODUCING.md` § "Appendix: reward-coefficient study" is
  updated to lock the values.
