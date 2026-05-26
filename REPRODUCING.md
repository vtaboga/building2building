<!-- -*- mode: markdown -*- -->

# REPRODUCING

This document maps every figure, table, and reported number in the
camera-ready Building2Building paper to the **exact command** that
regenerates it. Every entry has the same shape: *artefact → training /
eval command → plotting command → expected output path*.

> **Status (Phase B / C in flight).** The camera-ready paper replaces
> the legacy `task_const_e0`–`task_rand_e0` family with the normalized 3 × 3 family
> (`task_{const,occ,rand}_{e0,emed,ehigh}` — see
> `building2building/config/tasks.py`). Phase-C re-runs (PPO and SAC
> specialists, dynamics adaptation, cross-domain) are pending the
> Phase-B Slurm sweeps in `TODO.md`. Until those land, the commands
> below are the *intended* invocations; the artefacts they produce are
> what will populate the camera-ready figures and tables.

The only hard cross-phase dependency is **D2 (legacy reward deletion)
must precede C1 (regen `baseline_returns.csv`)** — see
`design_doc.md` § Roadmap.

---

## Conventions

- **Hydra entry points.** All training and reactive-control scripts
  live under `baselines/` and use Hydra. Config groups are resolved
  from `baselines/configs/`. Override any value from the CLI:

  ```bash
  python -m baselines.train_ppo experiment=train_ppo seed=0 \
      training.total_timesteps=5_000_000
  ```

- **Output directories.** By default Hydra writes to
  `outputs/${name}/${date}/${time}/` (see `baselines/configs/config.yaml`).
  Override with `output_dir=...`. SLURM submission scripts redirect to
  `$SCRATCH` to dodge the home-directory quota.

- **Slurm.** Long-running commands are packaged as `#SBATCH` scripts
  under `baselines/scripts/` and `analysis/task_study/sac_diagnostic/`.
  Per `AGENTS.md`, agents never `sbatch` themselves; the user submits.
  CPU-only nodes are sufficient for every job below — none of the
  baselines use GPUs.

- **Seeds.** The camera-ready specialist results use 3 seeds
  (`[0, 1, 2]`) per `(building_type, task, building)` cell. Use Hydra
  multirun to sweep:

  ```bash
  python -m baselines.train_sac experiment=train_sac_task_study \
      --multirun seed=0,1,2
  ```

- **Dataset.** Building data lives on HuggingFace
  (`vtaboga/building2building_dataset`) and is fetched on first use.
  No flag needed.

- **EnergyPlus.** EnergyPlus 24.1 must be on `PATH` or
  `$ENERGYPLUS_PATH`.

---

## Phase C deliverables

### C1. `baseline_returns.csv` (reactive controller)

The single CSV consumed by `b2b.compute_normalized_score` and the
plotting scripts to render the reactive-control reference line in
every figure.

**Train / eval (single-machine):**

```bash
python -m baselines.run_reactive_control experiment=eval_reactive_control \
    output_csv=building2building/scores/baseline_returns.csv
```

**Train / eval (Slurm sweep — recommended for the full test split,
~30 array tasks × 3 run periods):**

```bash
sbatch baselines/scripts/run_baseline_returns.sh
python baselines/scripts/merge_baseline_returns.py \
    "$SCRATCH/b2b_baseline_returns" \
    --output building2building/scores/baseline_returns.csv
```

**Plotting:** the CSV is referenced by every other figure; no plot
of its own.

**Artefact:** `building2building/scores/baseline_returns.csv`
(committed; row schema:
`building_type, building_id, task, run_period, reward_mean, ...`).

> **Camera-ready note.** The Slurm script currently iterates the
> legacy tasks. Replace them
> with the normalized presets — e.g.
> `task_const_e0`, `task_occ_e0`, `task_occ_emed`, `task_rand_e0`
> (or the full 9-cell grid if the paper switches to the full
> family) — and resubmit. This is the work-in-progress part of
> Phase D2 + C1 in `TODO.md`.

---

### C2. PPO specialists (paper §5)

Per-`(building_type, task, building)` PPO policy, 5 M steps, 14
parallel envs, 3 seeds.

**Train + eval (Hydra multirun across seeds, single-machine):**

```bash
python -m baselines.train_ppo experiment=train_ppo_task_study \
    --multirun seed=0,1,2
```

**Train + eval (Slurm array, one job per building):**

```bash
# 8 buildings per type × 1 type × 1 task per submission, 3 seeds via the loop:
for seed in 0 1 2; do
    sbatch baselines/scripts/train_ppo_small_test_array.sh \
        OfficeSmall 5000000 $seed
done
```

(`train_ppo_small_test_array.sh` currently hard-codes
`tasks=[task_const_e0]`; for the camera-ready run, swap to the normalized
preset(s) and rerun once per `(building_type, task)` pair.)

**Standalone evaluation (if re-evaluating saved checkpoints):**

```bash
python -m baselines.eval_ppo \
    --model-dir outputs/train_ppo_task_study \
    --output results_ppo_specialist.csv \
    --run-period full_year
```

**Plotting (Figure 4 / `fig_ppo_specialist`):**

```bash
python -m baselines.plotting.plot_ppo_specialist \
    --ppo-csv results_ppo_specialist.csv \
    --baseline-csv building2building/scores/baseline_returns.csv \
    --task task_occ_emed \
    --output figures/fig_ppo_specialist
```

(Re-invoke once per task to render every camera-ready panel.)

**Artefacts:**
- Models: `outputs/train_ppo_task_study/<seed>/models/<building_type>/<task>/ppo_<building_id>.zip`.
- CSV: `results_ppo_specialist.csv`.
- Figure: `figures/fig_ppo_specialist.{pdf,png}`.

---

### C2-bis. SAC specialists (paper §5, camera-ready addition)

Per-`(building_type, task, building)` SAC policy, 1 M steps, 4
parallel envs (off-policy: lower n_envs by design — see
`baselines/configs/training/sac.yaml`), 3 seeds.

**Train + eval (Hydra multirun across seeds, single-machine):**

```bash
python -m baselines.train_sac experiment=train_sac_task_study \
    --multirun seed=0,1,2
```

**Train + eval (Slurm array, one job per building × seed):**

```bash
for seed in 0 1 2; do
    SEED=$seed sbatch --array=0-7 baselines/scripts/train_sac_array.sh
done
```

(`train_sac_array.sh` parameterizes `BUILDING_TYPES`, `TASKS`,
`N_BUILDINGS` at the top of the file; edit those to cover the full
camera-ready grid.)

**Standalone evaluation:** SAC results are written by `train_sac.py`
itself to a per-run CSV; no separate `eval_sac.py` script exists.
The training-time CSV (`results.csv` under `output_dir`) is the
single source of truth for the post-training episode return and
normalized score.

**Plotting:** the same `plot_ppo_specialist` script can render SAC
once a `--sac-csv` argument is added; until then, hand-merge the SAC
training CSVs into a `results_sac_specialist.csv` with the same
schema as `results_ppo_specialist.csv` and pass them to a SAC-aware
variant of `plot_ppo_specialist`.

> **TODO before camera-ready submission.** Either (a) extend
> `plot_ppo_specialist.py` to accept a third bar group (sac), or
> (b) duplicate it as `plot_specialists.py`. Tracked in `TODO.md`
> as a follow-up to D5.

**Artefacts:**
- Models: `outputs/train_sac_task_study/<seed>/models/<building_type>/<task>/sac_<building_id>.zip`.
- CSV: `outputs/train_sac_task_study/<seed>/results.csv` (one row per
  trained policy; merge across seeds for the figure).

---

### C3. Dynamics adaptation (paper §6.1)

Three approaches at three difficulty levels:
- **specialist** — one PPO policy per training building (500 k steps,
  1 env each), evaluated on held-out buildings.
- **baseline** — one shared multi-building PPO policy (4 M steps, 16
  envs) without parameter augmentation.
- **parameterized** — same as baseline but with
  `AugmentObservationWithBuildingParams` (4 M steps, 16 envs).

**Train (per approach, per difficulty):**

```bash
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_specialist difficulty=easy

python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_baseline difficulty=easy

python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized difficulty=easy
```

(Repeat with `difficulty=medium` and `difficulty=hard` for the
3-panel figure. `easy` = SingleFamilyHouse, `medium` = OfficeSmall,
`hard` = OfficeMedium.)

**Evaluation (held-out test buildings):**

```bash
python -m baselines.eval_dynamics_adaptation \
    --model-path outputs/dynamics_specialist/models/specialist_*.zip \
    --difficulty easy --approach specialist \
    --output results_dynamics_specialist.csv

python -m baselines.eval_dynamics_adaptation \
    --model-path outputs/dynamics_baseline/models/multi_baseline.zip \
    --difficulty easy --approach baseline \
    --output results_dynamics_baseline.csv

python -m baselines.eval_dynamics_adaptation \
    --model-path outputs/dynamics_parameterized/models/multi_parameterized.zip \
    --difficulty easy --approach parameterized \
    --output results_dynamics_parameterized.csv
```

**Plotting (Figure 5a `fig_transfer_rew` + 5b `fig_transfer_temp_deviation`):**

```bash
python -m baselines.plotting.plot_dynamics_adaptation \
    --specialist-csv results_dynamics_specialist.csv \
    --baseline-csv   results_dynamics_baseline.csv \
    --parameterized-csv results_dynamics_parameterized.csv \
    --output figures/transfer/
```

**Artefacts:**
- `figures/transfer/model_comparison.png` (Fig 5a).
- `figures/transfer/temp_deviation_violin.png` (Fig 5b).

> **Camera-ready note.** `train_dynamics_*.yaml` currently uses
> `reward: task_const_e0`. Update to the chosen normalized preset (likely
> `task_occ_emed`) when D2 lands.

---

### C4. Cross-domain transfer / Amorpheus (paper §6.2)

Multi-building-type training with a heterogeneous transformer
policy.

**Train:**

```bash
python -m baselines.train_cross_domain experiment=train_cross_domain
```

**Evaluation (zero-shot on unseen building types):**

```bash
python -m baselines.eval_cross_domain \
    --model-path outputs/cross_domain/amorpheus_policy.pt \
    --test-building-types Warehouse SingleFamilyHouse \
    --task task_occ_emed \
    --n-test 20 \
    --output results_cross_domain.csv
```

**Plotting (`fig_cross_domain` / Figure 7):**

```bash
python -m baselines.plotting.plot_cross_domain \
    --results-csv results_cross_domain.csv \
    --output figures/transfer/comparison_figure
```

**Artefacts:**
- Model: `outputs/cross_domain/amorpheus_policy.pt`.
- CSV: `results_cross_domain.csv`.
- Figure: `figures/transfer/comparison_figure.png`.

> **Camera-ready note.** `train_cross_domain.yaml` currently
> defaults to `reward: task_occ_e0`; replace with the normalized
> equivalent when D2 lands.

---

## Phase B deliverables (calibration / stability)

These are not paper figures, but they generate the data committed
under `building2building/data/reward_normalizers.yaml` and
the SAC-stability table in `notes.md` § "SAC B2 ablation". Re-running
them is required if the calibration controller, run period, or SAC
config changes.

### B / A. Reward normalizers (calibration)

Computes the per-`(building_type, climate_zone)` `(τ_T, τ_E)`
constants under the SAC-warmup uniform-random reference controller.

```bash
python -m analysis.task_study.compute_random_policy_reward_normalizers \
    --mode aggregate
```

**Artefact:** `building2building/data/reward_normalizers.yaml`
(committed). Sanity plot lives under `analysis/` (gitignored;
location documented in `notes.md` § "Calibration sanity plot").

#### Partial regeneration (single building type)

When the pipeline changes the action space of one building type (e.g.
Phase M's OfficeMedium OA-mixer fix), only that type's
`(τ_T, τ_E)` rows need to be recalibrated.  The per-building rollout
cache under
`$SCRATCH/b2b_reward_normalizers_random/data/<run_period>/<bt>/`
is keyed by `(run_period, building_type, building_id)` only -- no
content hash -- so stale OfficeMedium caches would silently shadow
the new action space.  Invalidate and re-run:

```bash
# 1. Drop stale OfficeMedium caches.
rm -rf $SCRATCH/b2b_reward_normalizers_random/data/*/OfficeMedium/

# 2. Re-roll OfficeMedium across all 8 climate zones.  The launcher's
#    array indices 9..16 are the (OfficeMedium, CZ 1..8) buckets
#    (see analysis/task_study/scripts/launch_compute_random_policy_reward_normalizers.sh,
#    bucket index = type_index * 8 + (cz - 1), OfficeMedium is type_index 1).
sbatch --array=9-16 \
    analysis/task_study/scripts/launch_compute_random_policy_reward_normalizers.sh

# 3. After the array finishes, aggregate over ALL building types.
#    Non-OfficeMedium caches are unchanged, so their rows in the YAML
#    come out bit-identical to the pre-regen version; only the
#    OfficeMedium rows shift.
python -m analysis.task_study.compute_random_policy_reward_normalizers \
    --mode aggregate
```

Then commit the resulting `building2building/data/reward_normalizers.yaml`
diff (which `git diff` shows touches only the OfficeMedium `cz{1..8}`
rows across the three seasons).

### B2. SAC diagnostic ablation

5-cell × 3-seed × 6-building ablation on `task_occ_emed` /
`full_year` confirming that the three SAC fixes (`log_std_init=0`,
`ent_coef=0.2`, `use_sde=False`) restore stable learning.

```bash
sbatch --array=0-89 analysis/task_study/sac_diagnostic/submit_ablation.sh
```

**Artefact:** runs under `$SCRATCH/b2b_sac_diagnostic/cell{0..4}_*/seed{0..2}/`;
result table appended to `notes.md` § "SAC B2 ablation".

### B3. PPO `target_kl` re-tuning

Sweep `target_kl ∈ {None, 0.05, 0.1}` on `task_occ_emed` / full_year.
A small Hydra multirun is sufficient (no dedicated script):

```bash
python -m baselines.train_ppo experiment=train_ppo_task_study \
    --multirun \
    policy.target_kl=null,0.05,0.1 \
    seed=0,1,2 \
    "tasks=[task_occ_emed]"
```

**Artefact:** chosen winner committed to `baselines/configs/policy/ppo.yaml`;
note appended to `notes.md`.

---

## Reactive-controller tuning (paper §C / Appendix E.1)

The Optuna-tuned reactive controllers under
`baselines/configs/tuned_controllers/` are the reference for both
the reactive baseline and the calibration. To regenerate them
(e.g. after a building-pool change):

```bash
sbatch baselines/scripts/tune_controller.sh
```

This is a 41-element array sweep over `(building_type, climate_zone)`.
Outputs land under `$SCRATCH/b2b_tune_results/tuned_controllers/`;
copy the chosen YAMLs back into
`baselines/configs/tuned_controllers/` and commit.

For a single building / climate zone:

```bash
python -m baselines.tune_controller experiment=tune_controller \
    building_type=OfficeSmall climate_zone=1 reward=task_occ_emed
```

---

## PPO hyperparameter search (Appendix D.3)

The Patterson-et-al. CHS sweep for the PPO hyperparameters in
`baselines/configs/policy/ppo.yaml`. Not re-run every release;
included here for completeness.

```bash
# 75-worker sweep sharing one Orion DB
python -m baselines.tune_ppo experiment=tune_ppo task=task_occ_emed

# CHS analysis after the sweep finishes
python -m baselines.tune_ppo experiment=tune_ppo task=task_occ_emed analyze=true

# Final re-evaluation of the top configurations across many seeds
python -m baselines.tune_ppo experiment=tune_ppo task=task_occ_emed reeval=true
```

---

## Smoke tests

Before submitting any of the long jobs above, run the local smoke
tier to catch config regressions and import errors:

```bash
pytest -m quick
```

(Runs in ~5 min, no EnergyPlus required.)

For end-to-end EnergyPlus integration tests:

```bash
B2B_RUN_LONG_TESTS=1 pytest -m long
```

---

## Dataset regeneration

The dataset has two stages; each has its own entry point.

### Stage 1 — raw epJSON archive (run once per dataset version)

Applies Latin Hypercube Sampling over 7 envelope/geometry parameters to
the 16 ASHRAE 90.1-2022 prototype IDFs, producing the
`vtaboga/multizones_reference_buildings.zip` layout (6000 epJSONs +
`metadata.csv` + 16 EPWs).

**Single machine (all 6 types, 1000 samples each):**

```bash
python -m building2building.pipeline.generate_raw_dataset \
    --output-dir "$SCRATCH/b2b_raw_dataset" \
    --merge-metadata
```

**Slurm array (one task per building type, last task merges metadata):**

```bash
sbatch building2building/pipeline/scripts/generate_raw_dataset.sh
```

Stage 1 only needs to run if the LHS sampling, prototype IDFs, or
parameter ranges change.  For pipeline changes (actuator inventory,
schedules, HVAC control), run Stage 2 instead.

### Stage 2 — processed HF dataset (run whenever pipeline code changes)

Re-derives the controllable artefacts (`building.epjson`,
`equipment.json`, `metadata.json`) for the affected building types and
rewrites `metadata.parquet`.

**Single machine (slow but reproducible):**

```bash
python -m building2building.pipeline.generate_dataset \
    --building-type OfficeMedium \
    --output-dir "$SCRATCH/b2b_gen_dataset_OfficeMedium" \
    --write-metadata-parquet
```

**Slurm array (recommended — 20 shards × 50 buildings each):**

```bash
sbatch building2building/pipeline/scripts/generate_dataset.sh \
    --export=BUILDING_TYPE=OfficeMedium
```

> **Other building types.** Pass `--building-type` as a repeatable flag
> to regenerate `Warehouse`, `RetailStandalone`, `RestaurantFastFood`,
> and `OfficeSmall`.  `SingleFamilyHouse` has a different upstream source
> and is out of scope.

After all shards finish, the staging directory holds the per-building
artefacts plus the rewritten `metadata.parquet` and `splits.json`.
The current HF revision still uses the legacy per-type-zip layout
(`<BuildingType>.zip` at the repo root), so the upload step zips
the staging dir's `<BuildingType>/` tree and replaces the legacy zip:

```bash
cd "$SCRATCH/b2b_gen_dataset_<BuildingType>"

# Build the per-building-type zip from the staging tree.  Internal
# layout matches the legacy zip: top-level entries are
# <BuildingType>-NNNN/ (containing building.epjson, equipment.json,
# metadata.json, and the per-building EPW).
rm -f <BuildingType>.zip
(cd <BuildingType> && zip -rq ../<BuildingType>.zip .)

# Replace the legacy zip and refresh the unified metadata.parquet in
# a single commit.  Note: `huggingface-cli upload` is deprecated in
# huggingface_hub >= 1.14; use `hf upload` instead.
hf upload \
    vtaboga/building2building_dataset \
    . . \
    --repo-type dataset \
    --revision main \
    --include '<BuildingType>.zip' \
    --include 'metadata.parquet' \
    --include 'splits.json' \
    --commit-message "Regenerate <BuildingType> with post-M1 action space"
```

The package's pinned `REVISION = "main"` in
`building2building/data/download.py` then resolves to the new dataset
on first cache miss.  Users with a stale local cache will see a
loud `cattrs.ClassValidationError` on the missing `oa_mass_flow`
field — clearing
`~/.cache/huggingface/hub/datasets--vtaboga--building2building_dataset/`
fetches the new copy.  See `notes.md` § "OfficeMedium OA-mixer fix"
Q5 for the rationale.

---

## Cheat sheet

| Artefact | Command (single line, drop into a shell) |
|---|---|
| `baseline_returns.csv` | `sbatch baselines/scripts/run_baseline_returns.sh && python baselines/scripts/merge_baseline_returns.py "$SCRATCH/b2b_baseline_returns"` |
| PPO specialist (Fig 4) | `python -m baselines.train_ppo experiment=train_ppo_task_study --multirun seed=0,1,2` |
| SAC specialist (Fig 4) | `python -m baselines.train_sac experiment=train_sac_task_study --multirun seed=0,1,2` |
| Dynamics adaptation (Fig 5a/b) | `for d in easy medium hard; do for ap in specialist baseline parameterized; do python -m baselines.train_dynamics_adaptation experiment=train_dynamics_${ap} difficulty=${d}; done; done` |
| Cross-domain (Fig 7) | `python -m baselines.train_cross_domain experiment=train_cross_domain` |
| Reward normalizers | `python -m analysis.task_study.compute_random_policy_reward_normalizers --mode aggregate` |
| SAC B2 ablation | `sbatch --array=0-89 analysis/task_study/sac_diagnostic/submit_ablation.sh` |
| Reactive controller tuning | `sbatch baselines/scripts/tune_controller.sh` |
| HF dataset regen (Stage 2, OfficeMedium) | `sbatch building2building/pipeline/scripts/generate_dataset.sh --export=BUILDING_TYPE=OfficeMedium` |

---

## See also

- `README.md` — installation, public-API quick-start, smoke tests.
- `baselines/README.md` — config-by-config explanation of each
  Hydra group, plus the per-script CLI reference.
- `design_doc.md` — architectural and roadmap context.
- `notes.md` — in-flight decisions, gotchas, and result tables.
- `paper/main.tex` — authoritative figure / table definitions.
