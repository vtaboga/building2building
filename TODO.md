<!-- -*- mode: markdown -*- -->

# TODO

Atomic action items toward open-source release and paper rerun under the
new normalized reward. **One TODO, one commit** (per `AGENTS.md`).

## Execution order (2026-05-25 brainstorm)

The phases are listed in the order they must run end-to-end. Inside a
phase, items are reorderable except where a dependency is called out.

1. **Phase M — OfficeMedium OA-mixer action-space fix.** *Must land
   first.* Changes the action space → invalidates every OfficeMedium
   artefact downstream of it (HF dataset, tuned RBC, reward
   normalizers, baselines). Reference fix in `../RL2GNNs` (see M1).
2. **Phase D-house — codebase cleanup.** LICENSE, CI, legacy code
   purge, formatter passes, README refresh, **promote useful
   `analysis/` scripts into `baselines/`** (see "Cross-phase
   principles" below). Closes the OSS-release housekeeping debt.
3. **Phase T — test suite.** Per the principles already laid out in
   the Phase T preamble: pipeline coverage, contract tests, wrapper
   correctness, fixture matrix.
4. **Phase F — file and documentation audit (read-only).** Walk
   every file in `building2building/` and `baselines/` plus every
   docs page; produce a follow-up checklist. Includes the
   `analysis/` → `baselines/` migration audit. No edits in this
   phase.
5. **Phase R — empirical reward-coefficient study.** Determine the
   final values of `emed` and `ehigh` (and confirm `e0`) by
   training a SAC agent while monitoring the per-term reward
   distribution. Demonstrate (a) balance of the two terms at the
   chosen settings on a random → SAC-optimal policy ladder, and
   (b) invariance across buildings within `test_small`. The final
   plotting / data-gathering scripts live under `baselines/` (the
   prototype under `analysis/` is dev-only); the resulting
   figures are slated for the **paper appendix**.
6. **Phase C — paper rerun on the FINAL codebase.** Strict
   precondition: M, D-house, T, F, R must all be merged first so
   every Phase-C artefact is produced by the version of the code
   that will be open-sourced. Order inside C:
   - **C1.** Regen `baseline_returns.csv` on the **full
     train + test split** under the locked `emed`/`ehigh` values.
   - **C2.** PPO + SAC specialists on `test_small`, 1 seed,
     default params, full 9-task grid.
   - **C5 (partial).** Paper updates for the specialist figure
     and the task table.
   - **C3.** Dynamics adaptation (paper §6.1) — re-run as the
     final-step batch.
   - **C4.** Cross-domain Amorpheus (paper §6.2) — re-run as the
     final-step batch.
   - **C5 (final).** Paper updates for Fig 5 (dynamics) and
     Fig 7 (cross-domain).
7. **Phase D-API — public-surface finalization.** D5
   (`REPRODUCING.md`) is done; D9 (docs migration), D12 (contract
   marker), D13 (`CHANGELOG.md`), D14 (benchmark pages) close
   alongside / after Phase C.

**Hard dependency:** D2 (legacy reward deletion) was a hard
precondition for C1 in the previous plan. D2 is now ✓ done, so the
constraint is satisfied; the new hard constraint is **M completes
before C1** (the OfficeMedium row in `baseline_returns.csv` would
otherwise reference the old action space).

**Dropped:** B2 (SAC ablation), B3 (PPO `target_kl` re-tune), B4
(PPO reward-design full-year study). Decision 2026-05-25: no
further PPO/SAC hyperparameter sweeps before the camera-ready
rerun. The final agents use the existing defaults from B1
(`log_std_init=0.0`, `ent_coef=0.2`, `use_sde=false` for SAC;
`baselines/configs/policy/ppo.yaml` as committed for PPO).

---

## Cross-phase principles (2026-05-25)

These are project-wide rules that apply to every item below. They
modify or extend the principles in `AGENTS.md` and `design_doc.md`.

1. **`analysis/` is a development scratchpad. `baselines/` is the
   release surface.** Scripts that are intermediate exploration —
   one-off sanity plots, ablation harnesses used to make a decision
   that is then captured in a YAML — stay in `analysis/`. Any
   script whose **outputs are cited in the paper** (figures,
   tables, appendix plots) must have its final form — including the
   plotting code — committed under `baselines/`. The migration is
   tracked by **F2** (audit) and **D-house** (execution); it must
   be complete before Phase C runs, because Phase C runs against
   "the version of the code that will be open-sourced".

2. **`REPRODUCING.md` is kept in lock-step with the code.** Every
   item below that adds, moves, or removes a user-runnable script
   has an implicit "update `REPRODUCING.md`" sub-step. Specifically:
   when an item's acceptance criterion mentions a new entry
   point under `baselines/` or `building2building/`, the same
   commit must update the corresponding section of
   `REPRODUCING.md`. The cheat-sheet table at the bottom of
   `REPRODUCING.md` is the canonical reference; if a command is
   not in that table, it is not part of the reproduction path.

3. **The OSS release ships Python entry points, not Slurm
   scripts.** Slurm wrappers (`baselines/scripts/*.sh`,
   `analysis/task_study/scripts/*.sh`, `building2building/pipeline/
   scripts/*.sh`) are internal developer convenience for the
   compute cluster — they may live in the repo (and may be cited
   in `REPRODUCING.md` as "if you have Slurm, you can also run
   …"), but every paper artefact must be **reproducible from a
   `python -m …` invocation** that runs on a single machine. When
   a TODO item below proposes a Slurm script, the sibling
   `python -m …` invocation is the canonical entry point and the
   Slurm wrapper is a thin loop around it. If a script today only
   exists as a Slurm template, Phase D-house extracts the Python
   entry point.

4. **Decisions still owed (none for the current plan).** C3
   (dynamics adaptation) and C4 (cross-domain Amorpheus) are now
   confirmed in scope; they run as the final batch of Phase C
   per the user's 2026-05-25 follow-up. No `?`-marked items
   remain.

Format per item: short title; affected files; acceptance check;
references.

---

## Phase A — Finalize the reward ✓

Done. Calibration controller is the SAC-warmup uniform-random
policy; `reward_normalizers.yaml` is the locked YAML; sanity plot
regenerable via `python -m analysis.task_study.compute_random_policy_reward_normalizers --mode aggregate`.
See `notes.md` § "Reward — calibration regime and impl map" and
§ "Calibration sanity plot".

---

## Phase G — Reproducible dataset generation pipeline ✓

**Status (2026-05-26).** Complete. G1–G5 are committed and the new
OfficeMedium HF revision is live (`vtaboga/building2building_dataset`
at `26efedd9`, parent `ce0c68d9`). G5 follow-up commits this turn fixed
two latent bugs in G4 (missing per-building EPW emission;
`action_dim` counted the full E+ vector instead of the agent-facing
dim).

| Item | Code commit | Outstanding follow-up |
| --- | --- | --- |
| G1 — restore deleted IDF source as `building2building/sources/ashrae_90_1.py` | `3bfc4b8` | — |
| G2 — Stage 1 generator (`generate_raw_dataset.py` + Slurm wrapper) | `19fa101` | — |
| G3 — Stage 1 validation (one-off; **does not** replace the live HF zip) | `2e75673` (+ G3 closure commit this turn for `generate_raw_dataset.sh` arg fix and `tests/long/test_generate_raw_dataset_matches_existing.py` test redesign) | — (see § G3 closure) |
| G4 — Stage 2 generator (`generate_dataset.py`; replaces M2's `regen_dataset.py` + `regen_officemedium.sh`) | `098599c` (+ G5 follow-up commit for EPW emission, mem bump, and `agent_action_dim` helper) | — |
| G5 — Stage 2 validation + Phase M (M2) re-run | this turn's commit | — (HF revisions `ce0c68d9` (per-building artefacts) + `26efedd9` (corrected `metadata.parquet`); see § G5 post-mortem) |

**Two-stage architecture (mirrored in the file layout):**

```
Stage 1 (LHS sampling — runs once per dataset version)
  ASHRAE 90.1-2022 base IDFs (96) + 16 EPWs from
  energycodes.gov/ASHRAE901_all.zip
        + building2building/pipeline/generate_raw_dataset.py
        + Latin Hypercube Sampling over 7 envelope/geometry params
              ↓
  vtaboga/multizones_reference_buildings.zip (6000 epJSONs +
                                              metadata.csv + 16 EPWs)

Stage 2 (Pipeline processing — runs whenever pipeline code changes)
  multizones_reference_buildings.zip
        + building2building/pipeline/generate_dataset.py (Phase G4)
        + building2building/pipeline/_build_control_derivation
              ↓
  vtaboga/building2building_dataset    (per-building.epjson,
                                        equipment.json,
                                        metadata.json,
                                        metadata.parquet,
                                        splits.json)
```

**Why Phase G blocks Phase M (M2).** M2 was originally going to run
through `regen_dataset.py` + `regen_officemedium.sh`. Those two files
are scheduled for deletion in G4 (replaced by `generate_dataset.py`
+ `generate_dataset.sh`). G4's new entry point is a strict superset:
it accepts the same `--building-type OfficeMedium` flag, drops the
brittle hardcoded equipment-type counter in `rebuild_metadata_parquet`
in favour of counting actuators directly via `actuator_descriptions()`,
and adds a `--force` / skip-existing semantic that makes resumed Slurm
runs trivial. M2's acceptance criterion ("OfficeMedium HF dataset
regenerated under the post-M1 pipeline") is preserved verbatim;
**only the script path moves** (M2's `regen_officemedium.sh` ->
G4's `generate_dataset.sh --building-type OfficeMedium`).

### G1. Restore `building2building/sources/ashrae_90_1.py`

The Stage-1 generator needs read access to the 96 ASHRAE 90.1-2022
prototype IDFs (6 building types × 16 climate locations) plus the 16
TMY3 EPWs. The historical source `b2b/sources/energycodes.py` exposed
exactly this and was deleted in commit `1f57775` (April 2026
refactoring). G1 restores it as
`building2building/sources/ashrae_90_1.py`, modernised to the current
`store.py` API:

- `ASHRAE901_all_zip()`: a `DownloadFile` derivation pinned to the
  official URL
  `https://www.energycodes.gov/sites/default/files/2023-10/ASHRAE901_all.zip`
  with the same SHA-256 hash the historical module used
  (`de35252d...e212ab`). Re-download is content-verified.
- `ASHRAE901_all()`: the extracted-on-disk tree (via `ExtractZip`).
- `_index_buildings`, `_index_weathers`: `@derivation`-decorated
  parquet indices over the IDF and EPW filename conventions.
- `search_buildings(building_type=, year=, place=)` and
  `search_weathers(state=, filename=)`: thin DuckDB queries returning
  `pd.DataFrame` with a `path` column. Stage 1 reads IDF/EPW bytes
  directly off the content-hashed extracted tree, so unlike the
  historical version there is no `derivation` thunk column.

- Files: `building2building/sources/ashrae_90_1.py` (new).
- Acceptance: module imports cleanly; `ASHRAE901_all_zip()`,
  `ASHRAE901_all()`, `_index_buildings(...)`, `_index_weathers(...)`
  all construct without error and the `DownloadFile` carries the pinned
  hash. (Live download + index build is exercised in G2's smoke test;
  the G1 commit deliberately avoids triggering a 100 MB download
  during a normal `pytest` collection cycle.)

### G2. Stage 1 generator

`building2building/pipeline/generate_raw_dataset.py` (new) reproduces
the layout of `vtaboga/multizones_reference_buildings.zip` from the
G1 source. The historical `scripts/generate_dataset.py` (deleted in
`1f57775`) is the implementation reference; the modernised version
lives inside the package (per `AGENTS.md`'s "OSS release ships Python
entry points, not Slurm scripts" principle).

- Inputs (CLI):
  - `--output-dir <staging>` (required).
  - `--samples-per-type 1000` (matches existing zip).
  - `--seed 42` (matches existing zip).
  - `--building-type` (repeatable; defaults to all 6 of
    Warehouse, HotelSmall, RetailStandalone, RestaurantFastFood,
    OfficeMedium, OfficeSmall).
  - `--shard-index N --shard-count K` (Slurm parallelism; sharded by
    building type so shards write disjoint epJSON ID ranges).
  - `--force` (default off; skip-existing semantic on the per-building
    epJSON files).
- Output layout (matches existing zip):

      <staging>/
          metadata.csv                 (single file; full schema
                                        building_id, building_type, place,
                                        source_idf, weather_file, +
                                        the 7 LHS parameter columns)
          weather/*.epw                (16 files copied from the IDF
                                        zip's extracted tree)
          1.epJSON .. 6000.epJSON      (one per building, root-level)

- Per-type LHS loop (lifted from the historical script with no logic
  changes — the same `LatinHypercube(d=7, seed=42)` produces the same
  sample matrix):
  1. Resolve `(building_type, place)` rows via
     `search_buildings(building_type=bt, year=2022)`.
  2. Convert each base IDF to epJSON via
     `building2building.simulator.generator.convert_to_epjson`.
  3. For `i in range(samples_per_type)`: pick the base epJSON via
     `base_indices[i]`, deepcopy, apply
     `BuildingModification(**lhs_row)`, write
     `<staging>/{building_id}.epJSON`.
  4. Append a row to a per-shard partial `metadata_<shard>.csv`. After
     all shards finish, the last shard merges them into `metadata.csv`.
- A thin `building2building/pipeline/scripts/generate_raw_dataset.sh`
  Slurm wrapper handles the array submission (one task per building
  type, the last task does the merge).

- Files: `building2building/pipeline/generate_raw_dataset.py`,
  `building2building/pipeline/scripts/generate_raw_dataset.sh`,
  `REPRODUCING.md` § "Dataset regeneration" updated to document Stage 1.
- Acceptance: single-machine run with `--samples-per-type 1
  --building-type Warehouse` produces 1 epJSON, a 1-row
  `metadata.csv`, and the 16 EPW files; the row's
  `(building_type, place, source_idf, weather_file)` exactly match the
  upstream zip's row for `building_id=1`; the LHS parameter values
  match to float precision (LHS is deterministic given seed + d + n).

### G3. Stage 1 validation

A one-off validation that confirms Stage 1 actually reproduces the
existing `multizones_reference_buildings.zip`. **The existing live HF
artefact is NOT replaced** — Stage 1 just demonstrates it could be.

- Smoke test (committed under `tests/long/`, marked `@pytest.mark.long`
  so `pytest -m quick` skips it; runs a few minutes):
  `tests/long/test_generate_raw_dataset_matches_existing.py`:
  - Generate 1 building per type (`--samples-per-type 1`).
  - For each generated building, assert its row in the new
    `metadata.csv` is byte-equal (modulo float repr) to the
    corresponding row in the upstream zip's `metadata.csv`.
  - Round-trip the new epJSON through a 1-day E+ simulation; assert
    zero severe / fatal messages.
- Full metadata diff (exercised once, by the user, on the cluster):
  `sbatch building2building/pipeline/scripts/generate_raw_dataset.sh`
  with `--samples-per-type 1000` produces a 6000-row `metadata.csv`;
  the user diffs against the upstream zip's `metadata.csv`. The diff
  must be empty for the
  `(building_id, building_type, place, source_idf, weather_file)`
  columns; the 7 LHS parameter columns must match to at least 12
  decimal places (numpy LHS is deterministic but the dtype is
  float64).
- Per-type 5-sample E+ simulation (also under `tests/long/`):
  pick 5 random building IDs per type, run a 1-day E+ on each
  generated epJSON, assert all 30 sims complete with zero severe /
  fatal messages. Catches any regression in the IDF -> epJSON path.

- Acceptance: the smoke test passes locally; the user's full-grid
  diff is empty (or differs only on float-repr columns within 1e-12);
  the per-type E+ smoke completes 30/30.

#### G3 closure (2026-05-26) ✓

All three acceptance checks have now been exercised. The closure
commit lands two fixes that the validation surfaced:

1. **`building2building/pipeline/scripts/generate_raw_dataset.sh`:
   removed `--building-type "$BT"`.** The first cluster `sbatch` run
   failed in `main()` with
   `ValueError: --shard-count (6) must equal the number of requested
   building types (1)`. The wrapper was simultaneously narrowing
   `building_types` to one entry *and* claiming `--shard-count 6`; the
   Python entry point's invariant
   (`shard_count == len(building_types)`) correctly rejected this.
   The fix is to let `building_types` default to the canonical
   `ALL_BUILDING_TYPES` (length 6) and let `--shard-index` select the
   one type to process per array task. The shell script now has an
   explanatory comment in place pointing at
   `ALL_BUILDING_TYPES.index(bt)`.

2. **`tests/long/test_generate_raw_dataset_matches_existing.py`:
   rewrote both long tests around the `samples_per_type`-dependent
   shuffle in `generate_building_type`.** The two failures observed on
   the user's first cluster run were both *test design* issues, not
   pipeline regressions:

   - `test_generate_raw_dataset_smoke` previously joined the generated
     metadata against the upstream zip on `building_id`. But the
     per-type base-IDF shuffle in `generate_building_type` advances
     `np.random.default_rng(42)` by an amount that depends on
     `samples_per_type`, so the `building_id → place` mapping drifts
     between `samples_per_type=1` (the test) and
     `samples_per_type=1000` (the upstream zip). The test now joins
     on `(building_type, source_idf)` and asserts that the
     IDF → `(place, weather_file)` convention is consistent.
     This is the invariant Stage 1 actually needs to preserve; the
     `building_id → place` mapping is a downstream side-effect of the
     canonical samples_per_type choice.

   - `test_generate_raw_dataset_eplus_smoke` previously regenerated 5
     fresh epJSONs per type at `samples_per_type=5` and ran a 1-day
     E+ on each. Five of those 30 sims failed with
     `** Severe ** CheckWarmupConvergence: Zone "<…>" did not
     converge after 25 warmup days` (`ret=0`, no fatals — the sim
     completed). Reproducing locally:
       * The unmodified upstream Warehouse Denver IDF run through the
         same 1-day patched test passes cleanly (0 severes).
       * 10 different upstream-zip Warehouse Denver buildings (each
         with its own LHS sample) also pass cleanly.
       * Bumping the IDF's `MaximumNumberOfWarmupDays` to 50 does not
         clear the severes for our `bid=3` sample.

     Conclusion: warmup non-convergence is a thermal-physics property
     of the *specific* LHS 7-D parameter vector, not an IDF→epJSON
     regression. Because Stage 1's RNG state advances with
     `samples_per_type`, a fresh `samples_per_type=5` run produces
     LHS samples that simply do not appear in the upstream zip
     (which used `samples_per_type=1000`) — so they were never
     vetted by the upstream pipeline either. The test now samples
     5 buildings per type *from the upstream zip itself*, which is
     what Stage 1 reproduces byte-for-byte at the canonical
     `samples_per_type=1000`. The assertion is also tightened
     correctly: fail only on `ret != 0` or `** Fatal **`; `** Severe **`
     lines are logged via `print` (visible under `pytest -v -s`) but
     do not fail the test, since they don't indicate IDF→epJSON
     drift.

     Side effect: the test now uses `subprocess.run(<energyplus
     binary>, ...)` instead of `pyenergyplus.api.EnergyPlusAPI` so
     each E+ run gets its own process. The in-process API leaks
     cumulative state across runs (the same leak documented in the
     G5 post-mortem finding #2) and OOM-killed the test after ~16
     sims on a 4 GB box. Subprocessing bounds peak RSS to a single
     E+ process (~300 MB).

   Both tests now pass locally:
   `B2B_RUN_LONG_TESTS=1 pytest tests/long/test_generate_raw_dataset_matches_existing.py
   -m long -v -s` → `2 passed in 106.11s`, with 6 of 30 E+ sims
   logging informational severes (all `CheckWarmupConvergence`,
   `ret=0`).

3. **Full-grid metadata diff (the user's `sbatch` step).** Once the
   array job re-runs to completion with the corrected wrapper, the
   user's planned diff of `$SCRATCH/b2b_raw_dataset/metadata.csv`
   against the upstream zip's `metadata.csv` will be the third
   acceptance row. The metadata-smoke test above already exercises
   the IDF → `(place, weather_file)` convention; the full-grid diff
   additionally validates that the 7 LHS parameter columns are
   bit-identical (within float repr) at the canonical
   `samples_per_type=1000` — i.e. that the new pipeline byte-equals
   what was originally pushed to HF.

   No code follow-up is expected from the diff; if it does flag
   anything it would point at numpy LHS dtype drift between the old
   and new generators, which is unlikely given both call
   `scipy.stats.qmc.LatinHypercube` with `seed=42`.

#### G3 acceptance — all three rows green (2026-05-26)

Full-grid metadata diff was exercised by the user after the wrapper
fix landed (`sbatch` array job `9660823`, all 6 shards completed in
~9 min, merged into a 6000-row `metadata.csv` at
`$SCRATCH/b2b_raw_dataset/`).  The diff against the upstream zip
shows:

- **Discrete columns** (`building_id`, `building_type`, `place`,
  `source_idf`, `weather_file`): 0 mismatches across all 6000 rows
  (30 000 cells total).
- **LHS columns** (`envelope_conductivity_scale`, `window_u_factor`,
  `window_shgc`, `infiltration_scale`, `north_axis`, `scale_x`,
  `scale_y`): `max |Δ| = 0.0` exactly on all 7 parameters across all
  6000 rows — i.e. the CSV string representations are byte-identical,
  not merely numerically close.

This is a stronger result than the acceptance criterion required
(criterion was `1e-12`; we got bitwise zero). Stage 1 is therefore a
provably bit-perfect reproduction of upstream
`multizones_reference_buildings.zip`.

| Acceptance row | Status |
| --- | --- |
| Quick smoke (`test_generate_raw_dataset_smoke`) | ✓ PASS |
| Full-grid metadata diff (`sbatch` + upstream zip diff) | ✓ PASS (bitwise zero) |
| Per-type E+ smoke (`test_generate_raw_dataset_eplus_smoke`) | ✓ PASS (30/30, 0 fatals) |

G3 — and therefore Phase G as a whole — is closed. No HF re-upload
is planned (the existing zip stays canonical; Stage 1 has only
demonstrated that it *can* reproduce it).

### G4. Stage 2 generator (replaces M2's `regen_dataset.py`)

`building2building/pipeline/generate_dataset.py` (new) replaces
`building2building/pipeline/regen_dataset.py` and
`building2building/pipeline/scripts/regen_officemedium.sh`. M2's
acceptance criterion is preserved by routing the OfficeMedium re-run
through the new entry point.

- Building-list source: download `splits.json` from
  `vtaboga/building2building_dataset@main` (existing
  `building2building/data/download.py::download_splits`); take the
  union of `train ∪ test ∪ test_small` IDs for the requested building
  types. Single source of truth, no local `<BT>_*_data.json` files
  needed (those split files are not in the current tree anyway).
- Per-building loop:
  - Parse `processed_id = "OfficeMedium-4001"` -> `source_id = 4001`
    via `int(processed_id.rsplit("-", 1)[1])`. Fail loud on
    malformed IDs (no fallback to reading the existing HF cache).
  - Skip if all three artefact files (`building.epjson`,
    `equipment.json`, `metadata.json`) already exist under
    `<output_dir>/<building_type>/<processed_id>/` and `--force` is
    not set.
  - Otherwise call
    `building2building.sources.multizones_reference_buildings.\
    _build_control_derivation(zip, f"{source_id}.epJSON",
    "full_year")` (existing function; Stage 2 just drives it).
  - Run `extract_discovery_metadata(...)` to compute
    `net_conditioned_area` and `warmup_phases`. **No shortcut
    --rerun-discovery flag** (always runs from scratch).
  - Write the three artefacts.
- `metadata.parquet` rebuild rewrites the `action_dim` column for the
  regenerated rows by counting actuators directly from the
  `equipment_list` returned by `_build_control_derivation` (i.e.
  `sum(len(e.actuator_descriptions()) for e in equipment_list)`),
  rather than re-parsing `equipment.json` via the brittle hardcoded
  per-equipment-type counter currently in
  `regen_dataset.rebuild_metadata_parquet`. Other columns
  (`num_zones`, `observation_dim`, `weather_file`, `hvac_type`,
  `climate_zone`) are copied from the upstream parquet unchanged for
  non-regenerated rows; for regenerated rows the per-building
  summary dict carries them through. **No silent skip** for missing
  per-building dirs (current M2 logic warns and continues; new logic
  raises).
- `splits.json`: copied unchanged from the existing HF dataset (we
  are keeping the same buildings — see Phase M Q5).
- CLI: `python -m building2building.pipeline.generate_dataset
  --output-dir <staging> [--building-type <bt> ...]
  [--shard-index N --shard-count K] [--force]
  [--write-metadata-parquet]`. Slurm wrapper at
  `building2building/pipeline/scripts/generate_dataset.sh` (generic
  over `--building-type`).
- **Deletions in the same commit:**
  - `building2building/pipeline/regen_dataset.py`.
  - `building2building/pipeline/scripts/regen_officemedium.sh`.
  - `REPRODUCING.md` § "Dataset regeneration" rewritten to point at
    the new entry points.

- Acceptance: smoke test (committed under `tests/long/` with skip-if-no-HF
  marker) regenerates 1 OfficeMedium building, asserts the new
  `equipment.json` round-trips through `cattrs.structure`,
  `action_dim` in the rewritten parquet matches
  `len(equipment_list[*].actuator_descriptions())`, and `splits.json`
  is bit-identical to the upstream copy.

### G5. Stage 2 validation + M2 re-run ✓

Done. The user re-ran the Slurm wrapper for OfficeMedium and uploaded
the staging dir to HF. This subsumes Phase M's M2.

- HF revisions:
  - `ce0c68d9` — per-building artefacts re-derived through the
    post-M1 pipeline (replaces legacy `OfficeMedium.zip`; drops the
    broken flat `OfficeMedium/` tree pushed in interim `8768bd82`).
  - `26efedd9` — `metadata.parquet` corrected so `action_dim` is the
    agent-facing dim (36 for OfficeMedium), not the full E+ dim
    (51); other 5 building types unchanged from the upstream parquet.
- Slurm runs:
  - `sbatch ... generate_dataset.sh --export=BUILDING_TYPE=OfficeMedium`
    @ job `9655412` (20 shards × 16 GB; 5–8 min per shard).
  - Followed by a local `--shard-count 1 --write-metadata-parquet`
    invocation (~3 s) to rewrite `metadata.parquet` with the
    `agent_action_dim` helper landed in this turn.
- Push command (canonical, replaces the deprecated
  `huggingface-cli`):
  - `hf upload vtaboga/building2building_dataset . . --repo-type
    dataset --revision main --include 'OfficeMedium.zip' --delete
    'OfficeMedium/**'` for `ce0c68d9`.
  - `hf upload vtaboga/building2building_dataset metadata.parquet
    metadata.parquet --repo-type dataset --revision main` for
    `26efedd9`.

- Acceptance (verified this turn): for `k=0` in `splits["train"]`
  of all 6 building types,
  `b2b.new_make_env(bt, split="train", index=k).action_space.shape[0]`
  equals `registry.get_building_by_id(...).action_dim`:

      OfficeMedium       36 (= 51 raw − 15 fixed cooling SP)
      OfficeSmall        10
      Warehouse           5
      RetailStandalone    9
      RestaurantFastFood  4
      SingleFamilyHouse   2

- Phase M (M2) status updated to "completed via G5 in HF revisions
  `ce0c68d9` + `26efedd9`".

#### G5 post-mortem (2026-05-26)

G4's first end-to-end run surfaced four latent bugs in
`generate_dataset.py` and `generate_dataset.sh`, all fixed in this
turn's commit:

1. **Missing per-building EPW emission.** `generate_one_building`
   wrote `building.epjson` + `equipment.json` + `metadata.json` but
   not the EPW, even though `factory.py:55` and `api/__init__.py:294`
   both resolve `weather_path = info.building_dir / info.weather_file`.
   The legacy zip layout (preserved by the post-G5 fix) carries the
   EPW under `<store-hash>-<basename>.epw` per
   `store.py::realize` (line 73). Fix: `shutil.copy(epw_path,
   target_dir / epw_path.name)` after `realize(STORE_PATH.get(),
   epw_derivation)`. The skip-existing check in
   `generate_building_type` was extended to also require `*.epw`.
2. **`--mem=8G` is too tight for the per-shard loop.** OOM-killed
   shards 18 and 19 at item 46/50 after ~8 minutes wall-clock.
   Symptom is a slow per-building memory leak in
   `_build_control_derivation`/`realize` (the store grows
   monotonically; 50 buildings × ~150 MB of intermediate artefacts).
   Bumping to 16 GB unblocked the run; the leak itself remains
   uninvestigated (filed as a TODO comment in `generate_dataset.sh`).
3. **`action_dim` counted the full E+ vector, not the agent-facing
   dim.** G4's design (line 339-345) specified
   `sum(len(e.actuator_descriptions()) for e in equipment_list)`,
   which yields 51 for OfficeMedium (3 SAT + 45 terminal + 3 OA).
   But the simulator filters 15 cooling-setpoint actuators via
   `_is_fixed_actuator` and pins them at 40 °C, so the agent-facing
   `env.action_space.shape[0]` is 36. Downstream consumers that
   trust `BuildingInfo.action_dim` (e.g.
   `benchmarks/dynamics_adaptation.py`, future policy-network sizing
   from the parquet) would have allocated 15 dead slots. Fix: new
   public helper `building2building.simulator.action_spaces.\
   agent_action_dim` reuses the same `hvac_action_space` filter the
   simulator uses at runtime; `rebuild_metadata_parquet` routes
   through it. This turn's commit also updates
   `benchmarks/dynamics_adaptation.py:19` (`"hard": 33` → `36`).
4. **`huggingface-cli` is deprecated in `huggingface_hub ≥ 1.14`.**
   `REPRODUCING.md` § "Dataset regeneration" still pinned the old
   command; the user hit a hard "deprecated and no longer works"
   error. Replaced with `hf upload` in both `REPRODUCING.md` and
   above.

A latent cleanup spotted in `rebuild_metadata_parquet` while fixing
(3): the pre-fix code dispatched cattrs structuring per
`equipment_type` and checked `et == "heatingonlyzone"`, but the
actual literal is `"heating_only"` (`actuators.py:646`). This would
have raised `ValueError: Unknown equipment_type 'heating_only'` on
any building with a heating-only zone (OfficeMedium has none, so
this never fired in G5). The new code structures via the
`AnyEquipment` discriminated union and drops the per-type
dispatch.

---

## Phase M — OfficeMedium OA-mixer action-space fix ≈

**Status (2026-05-25).** Code-side complete on branch
`phase/m-officemedium-action-space` (not yet pushed). Each item below
has a committed code/docs landing; the artefact updates that require
Slurm runs are tracked as outstanding follow-up commits.

| Item | Code commit | Outstanding follow-up |
| --- | --- | --- |
| M0 — design note | `3c84984` | — |
| M1 — pipeline change | `5a9772d` | — |
| M2 — HF dataset regen | `28f2e0a` (interim — superseded by Phase G4 `098599c`) | **completed via G5** (HF revisions `ce0c68d9` + `26efedd9`). `regen_dataset.py` + `regen_officemedium.sh` are deleted; the canonical entry point is `sbatch building2building/pipeline/scripts/generate_dataset.sh --export=BUILDING_TYPE=OfficeMedium`. See § G5 post-mortem for the four bugs that surfaced and were fixed during validation. |
| M3 — RBC pins OA + retune | `0141961` | user runs `sbatch baselines/scripts/tune_controller.sh`, then commits the 8 new `air_loop_officemedium_cz{1..8}.yaml` |
| M4 — reward normalizers | `e9da856` (docs-only, OfficeMedium-partial-regen plan) + this turn's commit (calibration-script fix: drop `task3` reference and the reward-arithmetic `temp_penalty` recovery — both invalid post-D2 — and read components directly from `info["raw_observation"]` via `_deadband_components`; consequence: full regen, not just OfficeMedium) | user runs `rm -rf $SCRATCH/b2b_reward_normalizers_random/data/` + `sbatch .../launch_compute_random_policy_reward_normalizers.sh` (full 1-48 array) + `--mode aggregate`, then commits the resulting `reward_normalizers.yaml` diff (touches every row) |

**This phase blocks every other downstream artefact.** Run first.

### Context

The current OfficeMedium action space, produced by
`building2building/pipeline/.../make_controllable`, is missing the
**outdoor-air mixer** actuator (a.k.a. *OA mixer* / *outdoor-air
controller*). Without it, the agent cannot regulate the
mixed-air fraction at the air-handling unit, which is the dominant
energy-versus-IAQ trade-off in any VAV/AHU building. The
consequence is that no control policy — RBC, PPO, SAC — can
reach a meaningful optimum on OfficeMedium; the resulting numbers
in the paper are not representative of the achievable best on the
real building.

A working fix already exists in the sibling cluster repo
`../RL2GNNs` and runs on **one** instance of OfficeMedium there
(see `../RL2GNNs/officerl/action_spaces.py`,
`../RL2GNNs/officerl/graph.py` line ~500, and
`../RL2GNNs/officerl/data/building.epjson`). The relevant
actuator entries are:

- `(component_type="Outdoor Air Controller", control_type="Air Mass Flow Rate")`
  — one per air loop. This is the OA-flow command exposed as an
  agent action in RL2GNNs.
- `(component_type="AirLoopHVAC", control_type="Availability Status")`
  — one per air loop. Availability of the air handler.

B2B's pipeline currently emits neither for OfficeMedium. The
RL2GNNs reactive baseline holds the OA mixer at a constant value;
the user-confirmed plan is to do the same in B2B's reactive
controller, but **expose the OA mixer command as part of the
agent's action space** so PPO/SAC can learn to use it.

### Downstream invalidations once the action space changes

The new action set forces a re-roll of every OfficeMedium-touching
artefact:

| Artefact | Where | Re-roll item |
| --- | --- | --- |
| HuggingFace dataset (every OfficeMedium epJSON / equipment.json / metadata row) | `vtaboga/building2building_dataset` | **M2** |
| Tuned reactive controller for every OfficeMedium climate zone | `baselines/configs/tuned_controllers/air_loop_officemedium_cz{1..8}.yaml` | **M3** |
| Reward normalizers `(τ_T, τ_E)` for every OfficeMedium climate zone | `building2building/data/reward_normalizers.yaml` (OfficeMedium rows) | **M4** |
| `baseline_returns.csv` rows for OfficeMedium | `building2building/scores/baseline_returns.csv` | Subsumed by **C1** |

Items M2–M4 are unavoidable once M1 lands; they cannot be skipped
or partially run.

### M0. Brainstorm — specifics document

**Before any code change.** Produce a short design note (~1 page,
committed to `notes.md` § "OfficeMedium OA-mixer fix") that pins
the design choices below by reading the RL2GNNs reference and
checking the assumptions hold for **all 1000 OfficeMedium
buildings** (not just the one RL2GNNs uses).

Questions to answer in the brainstorm:

1. **Per-air-loop multiplicity.** OfficeMedium has multiple air
   loops (3 in the prototype, see `VAV_1`, `VAV_2`, `VAV_3` in
   the RL2GNNs epJSON). Does every one of the 1000 OfficeMedium
   instances in our HF dataset have the same 3-air-loop topology,
   or do some variants exist? Answer impacts how the actuator
   list is enumerated.
2. **Actuator scheme: agent or RBC?** RL2GNNs exposes the OA
   mixer to the agent. User confirmed B2B does the same. Confirm
   the action range and units against EnergyPlus's
   `Air Mass Flow Rate` actuator (kg/s). Decide whether to also
   expose `AirLoopHVAC × Availability Status` as a discrete
   action or keep it pinned `On` (RL2GNNs exposes it; for the
   B2B paper a binary availability action complicates the
   continuous action space — recommended: **pin to `On` for B2B,
   diverging from RL2GNNs**).
3. **Reactive controller constant value.** What value does the
   RL2GNNs RBC pin the OA mixer to? Is it autosize, minimum-OA,
   a fraction of design flow, or zone-occupancy-modulated? Pin
   the same scheme in B2B's reactive controller, and document
   the exact value in the new tuned-RBC YAMLs.
4. **Pipeline change locus.** Is this a change to
   `building2building/pipeline/make_controllable.py` (add the OA
   mixer to the actuator-set output) only, or does
   `extract_discovery_metadata` also need to learn about the new
   actuator type? Verify with one local pipeline dry-run before
   M1 commits.
5. **Backward compat.** Buildings already downloaded by users
   from the **old** HF revision must still load (with the old
   action space) or fail with a clear error. Recommended:
   bump the dataset's revision tag (e.g. `v0.2.0` →
   `v0.3.0`) and have `building2building/data/registry.py`
   pin the new revision in code so a `pip install -U` user
   gets the new dataset automatically.

- Files: `notes.md` § "OfficeMedium OA-mixer fix" (new section).
- Acceptance: design note merged; every numbered question above
  has an answer; M1 cannot start until this commit lands.

### M1. Pipeline change: emit OA-mixer actuator for OfficeMedium

After M0 locks the design. Touch the smallest possible surface
that produces the right `ActuatorDescription` list for
OfficeMedium.

- Files: `building2building/pipeline/make_controllable.py` (or
  whichever module emits the actuator list — verify in M0); the
  per-HVAC actuator template under `building2building/pipeline/`;
  one local test against a hand-carved OfficeMedium fixture.
- Acceptance: running `make_controllable` on **one** real
  OfficeMedium epJSON pulled from the HF dataset produces an
  actuator list that contains exactly the new
  `Outdoor Air Controller × Air Mass Flow Rate` entries (one per
  air loop) in addition to the existing actuators; the rest of
  the actuator list is unchanged. Diff against RL2GNNs's
  expected actuator list for the same building is empty (modulo
  the M0-confirmed divergence on `Availability Status`).
- Add a `tests/quick/test_officemedium_actuator_set.py` regression
  test fixed against a committed minimal OfficeMedium fixture
  (T0a's `minimal_vav/`, which is an OfficeMedium carve-out) so
  the actuator set is pinned by the test suite.

### M2. Regenerate the HuggingFace dataset for OfficeMedium

The 1000 OfficeMedium epJSONs + `equipment.json` files must be
regenerated through the post-M1 pipeline and pushed.

- Files (input): the source IDFs for OfficeMedium under whatever
  staging directory the dataset was built from originally (the
  brainstorm M0 needs to locate this). The B2B repo itself
  doesn't store these.
- Files (output): the `OfficeMedium` slice of
  `vtaboga/building2building_dataset` (revision bump per M0).
- Release entry point (per Cross-phase principle 3): the dataset
  regeneration is a Python script under
  `building2building/pipeline/regen_dataset.py` (new — or extend
  an existing module if one exists; verify in M0). Canonical
  invocation:

      python -m building2building.pipeline.regen_dataset \
          --building-type OfficeMedium --output-dir <staging>

  The single-machine command must work for the full 1000-building
  loop (slow but reproducible). A thin Slurm wrapper
  `building2building/pipeline/scripts/regen_officemedium.sh`
  parallelizes the Python entry point across `--array=0-999` for
  cluster runs but is **not** the canonical reproduction path.
- Acceptance: HF dataset revision bump; every OfficeMedium row in
  the metadata parquet shows the new actuator-count field;
  `b2b.new_make_env("OfficeMedium", split="train", index=k)` loads
  successfully for all `k` in the new test_small subset and the
  env's `action_space.shape[0]` reflects the new actuator count.

### M3. Re-tune the Optuna reactive controller for OfficeMedium

The reactive baseline parameters in
`baselines/configs/tuned_controllers/air_loop_officemedium_cz{1..8}.yaml`
were tuned against the **old** action space and are now stale.

- Release entry point (per Cross-phase principle 3): the existing
  `baselines/tune_controller.py` Hydra script. Canonical
  invocation (per CZ):

      python -m baselines.tune_controller experiment=tune_controller \
          building_type=OfficeMedium climate_zone=<n> reward=task_occ_emed

  The Slurm wrapper `baselines/scripts/tune_controller.sh` loops
  this across the 8 OfficeMedium climate zones for cluster runs.
- Files: regenerate all 8 OfficeMedium climate-zone YAMLs in
  `baselines/configs/tuned_controllers/`. The OA-mixer command
  in the RBC is held at the constant value pinned by M0 (not
  Optuna-searched); only the existing knobs (deadband widths,
  AHU schedule offsets, etc.) are re-searched.
- Acceptance: 8 new `air_loop_officemedium_cz*.yaml` files
  committed; previous versions overwritten (no `_v2` files left
  around — per `AGENTS.md` "track via git, not via script
  duplication"); `REPRODUCING.md` § "Reactive-controller tuning"
  unchanged (it already cites the Python entry point).

### M4. Re-run reward-normalizer calibration (all building types)

The `(τ_T, τ_E)` constants for OfficeMedium in
`building2building/data/reward_normalizers.yaml` were computed
under the old action space. The temperature term shouldn't shift
much (zones are the same) but the power term will — the OA mixer
materially changes the energy use under any non-trivial control
policy. Recompute to keep the per-bucket calibration honest.

**Scope upgrade (2026-05-26):** auditing the calibration scripts
during M4 prep revealed that both
`analysis/task_study/compute_random_policy_reward_normalizers.py`
and `compute_reward_normalizers.py` hard-coded the legacy preset
`task3` (deleted in D2) and recovered `temp_penalty` by inverting
the un-normalized deadband formula
(`temp_penalty = -reward - w_E * power_penalty`), which is
invalid against the only surviving reward
`NormalizedDeadbandReward`. The scripts now pass
`task_occ_emed` (semantic successor) and recompute components
directly from `info["raw_observation"]` via
`_deadband_components`, using the comfort zones / target schedule
/ `dT` from the live `env.unwrapped.reward_fn`. Consequence: M4
is a **full regen** of every row, not just OfficeMedium —
non-OfficeMedium rows in the committed YAML were also computed
via the now-known-buggy path and should not be carried forward.

- Release entry point: the calibration is run via the existing
  `analysis/task_study/compute_random_policy_reward_normalizers`
  module. Note that this module is one of the **(promote to
  baselines/)** candidates flagged in F2 — its outputs feed
  `reward_normalizers.yaml` which ships with the package, so the
  final version of the script should live under `baselines/`. If
  D15 has already moved it, use the new path; if not, M4
  triggers the move (file an FX sub-commit). Canonical
  invocation post-move:

      python -m baselines.compute_reward_normalizers \
          --mode aggregate --building-type OfficeMedium

  (the exact CLI is up to the F2/D15 audit; M0 should pin it).
- Files: merge the new OfficeMedium rows into
  `building2building/data/reward_normalizers.yaml`; leave the
  other building types' rows untouched. Update `REPRODUCING.md`
  § B/A if the entry point has moved.
- Acceptance: the OfficeMedium rows in `reward_normalizers.yaml`
  are updated; the rest of the file is bit-identical to the
  pre-M4 version (`git diff` shows only OfficeMedium changes); the
  sanity plot regenerated by the Python invocation above shows
  medians at 1.0 for OfficeMedium under the new normalizer;
  `REPRODUCING.md` cites the (possibly new) entry-point path.

---

## Phase B — Stabilize RL training under the new reward ✓ / cancelled

- **B0 + B0.1 (EnergyPlus leak fix).** ✓ Done. Upstream commits
  `6d03b9a` + `956c3e1` on `vtaboga/minergym`, pinned in
  `pyproject.toml`; in-tree `B2BEnergyPlusEnvironment` subclass
  removed; `tests/long/test_env_leak.py` is the regression guard.
  Residual ~14 MB/cycle native RSS growth is irreducible from
  Python (would need subprocess isolation — see `minergym_todo.md`
  for the deferred spec). Operational details + future-failure
  watchlist live in `notes.md` § "Operational gotchas".
- **B1 (SAC policy config: `log_std_init=0.0`, `ent_coef=0.2`,
  `use_sde=false`).** ✓ Done. Commit `64a4eb5` on
  `feature/reward-normalization`. Rationale in `notes.md` § "SAC".
- **B2, B3, B4.** Cancelled 2026-05-25. No further PPO/SAC
  hyperparameter sweeps before the camera-ready rerun; the final
  agents use B1's SAC defaults and the committed
  `baselines/configs/policy/ppo.yaml`. B4's "balance at 1.0"
  validation is **superseded by Phase R** (which trains a SAC
  agent on `full_year` while monitoring per-term reward
  contributions on `test_small`).

---

## Phase C — Re-run paper experiments (FINAL stage, on the OSS codebase)

**Strict prerequisites — all of these must be merged before C1
starts:**

1. **Phase M** (OfficeMedium fix + HF re-upload + RBC re-tune +
   normalizer recompute) — otherwise OfficeMedium rows are
   produced against the wrong action space.
2. **Phase D-house** (legacy deletion, LICENSE, formatter,
   `pyright`, README refresh).
3. **Phase T** (test suite green on `pytest -m quick`).
4. **Phase F** (read-only file/doc audit; follow-up edits from F
   landed in D-house or merged separately before C starts).
5. **Phase R** (final `emed` / `ehigh` values pinned in
   `building2building/config/tasks.py` so every Phase-C run uses
   the locked coefficients).

The intent is captured by the user's "the scripts to generate the
new results must be run on the final version of the codebase"
rule: every artefact produced in Phase C must be reproducible by
the public OSS code, not by the in-flight branch.

### C1. Regenerate `baseline_returns.csv` on the **full** split

Run `baselines/run_reactive_control.py` across **every**
`(building_type, building_id, task, run_period)` tuple in the
full train + test split, under the locked `emed` / `ehigh`
values from Phase R. The reactive baseline is the reference
controller for the normalized score and underlies every figure
in the paper — both the specialists (C2) and the transfer
benchmarks (C3 / C4), so train-split rows are not optional.

- Files (release artefact): `building2building/scores/
  baseline_returns.csv` (replaced in full); schema unchanged from
  the existing header.
- Files (release entry point): `baselines/run_reactive_control.py`
  + the existing
  `baselines/configs/experiment/eval_reactive_control.yaml`. The
  full reproduction command (in `REPRODUCING.md` § C1) is the
  single-machine Hydra invocation:

      python -m baselines.run_reactive_control \
          experiment=eval_reactive_control \
          output_csv=building2building/scores/baseline_returns.csv

- Files (developer convenience): `baselines/scripts/
  run_baseline_returns.sh` Slurm wrapper for the cluster (does
  **not** ship as the canonical reproduction step per
  Cross-phase principle 3). Update the Slurm script's header to
  iterate the full 9-cell task grid + 3 run periods + the full
  building catalogue; user submits.
- Tasks × run periods: full 9-cell grid
  (`task_{const,occ,rand}_{e0,emed,ehigh}`) × `run_period ∈
  {full_year, winter, summer}`. Total row count ≈
  `len(full_split) × 9 × 3`.
- Acceptance: every `(building_type, building_id, task,
  run_period)` tuple the paper figures cite is present;
  `pytest -m quick tests/quick/test_scoring.py` passes;
  `tests/release/test_baseline_returns_coverage.py` (T16) passes
  against the new CSV; `REPRODUCING.md` § C1 is updated to point
  at the locked-`emed`/`ehigh` values.

### C2. PPO + SAC specialists on `test_small` (paper §5)

Scope per the 2026-05-25 brainstorm: **1 seed, default params, no
tuning, full 9-task grid, evaluated on the test_small subset of
every building type.**

Total cells = `len(building_types) × len(test_small per type) ×
9 tasks × 2 algorithms × 1 seed`. With 6 building types and ~5
test_small buildings per type, that is `6 × 5 × 9 × 2 = 540`
training runs. Each PPO specialist run is ~5M steps × 14 envs;
each SAC specialist run is ~1M steps × 4 envs. CPU-only.

The reduction from the camera-ready original plan (3 seeds, full
test split) to (1 seed, test_small) is deliberate compute
shedding. **Document this reduction in the paper prose
(`paper/main.tex` §5)** as part of C5a — the figure caption must
state the seed count and the building subset.

- Files (release artefacts): `baselines/train_ppo.py`,
  `baselines/train_sac.py` Hydra outputs collected under
  `outputs/`; merged CSVs `results_ppo_specialist.csv` +
  `results_sac_specialist.csv`; combined PPO+SAC specialist
  figure rendered by an extended
  `baselines/plotting/plot_ppo_specialist.py` (or a new
  `baselines/plotting/plot_specialists.py` covering both
  algorithms — pick one in the C2 first commit, per the D5
  follow-up noted in `REPRODUCING.md`).
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
  first C2 commit if missing.
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
  explicit in the figure caption; `REPRODUCING.md` § C2 + § C2-bis
  are updated to reflect the locked `emed`/`ehigh` values and the
  combined-figure command.

### C5a. Specialist paper updates

Land **after C2** so the specialist figure exists. Splits C5 into
two; C5a covers everything that depends only on C1+C2; C5b
covers C3/C4.

- Files: `paper/main.tex` —
  - Task table: replace any remaining legacy `Task 1`–`Task 4`
    references with the normalized 9-cell
    `task_{const,occ,rand}_{e0,emed,ehigh}` family. Pin the
    final `emed` and `ehigh` values from Phase R in the table
    caption.
  - Swap Fig 4 (PPO specialist) for the new PPO + SAC side-by-
    side figure from C2.
  - Update §5 prose: seed count (1 not 3), building subset
    (`test_small`), per-term reward decomposition citation
    forward-referencing the appendix from Phase R.
  - Address the reviewer-flagged confound about cross-building
    `w_E` comparability explicitly in §5 prose; cite the Phase R
    invariance plot in the appendix.
  - **Add the Phase R appendix.** Per the user's 2026-05-25
    follow-up, the empirical reward-coefficient study is included
    in the camera-ready as an appendix. The appendix lays out R0's
    methodology, the policy ladder, and the invariance result;
    figures from R3 are embedded.
- Acceptance: PDF compiles; reviewer-flagged confound is
  addressed; the Phase R appendix is present and cited from §5;
  user has reviewed and approved the diff.

### C3. Dynamics adaptation (paper §6.1) — final step

Confirmed in scope (2026-05-25 follow-up). Runs as the
**final-step batch** after C1, C2, C5a are complete, on exactly
the same OSS-ready codebase.

Re-run the three approaches (specialist, baseline, parameterized)
at three difficulty levels (easy = SingleFamilyHouse, medium =
OfficeSmall, hard = OfficeMedium) under the new reward and the
new OfficeMedium action space. 1 seed, default params, matching
the C2 compute-shedding rationale.

- Files (release entry points, cited from `REPRODUCING.md` § C3):

      python -m baselines.train_dynamics_adaptation \
          experiment=train_dynamics_specialist difficulty=easy
      python -m baselines.train_dynamics_adaptation \
          experiment=train_dynamics_baseline difficulty=easy
      python -m baselines.train_dynamics_adaptation \
          experiment=train_dynamics_parameterized difficulty=easy

  (Repeat with `difficulty=medium,hard`.) Plotting via
  `baselines.plotting.plot_dynamics_adaptation` — see
  `REPRODUCING.md` § C3 for the full invocation.
- Acceptance: `fig_transfer_rew` and `fig_transfer_temp_deviation`
  regenerate from a single Python command per
  `REPRODUCING.md`; seed count and OfficeMedium action-space
  update are noted in the figure caption.

### C4. Cross-domain Amorpheus (paper §6.2) — final step

Confirmed in scope (2026-05-25 follow-up). Runs alongside C3 as
part of the final-step batch.

- Files (release entry points, cited from `REPRODUCING.md` § C4):

      python -m baselines.train_cross_domain experiment=train_cross_domain

  followed by the eval and plotting commands documented in
  `REPRODUCING.md` § C4.
- Acceptance: the cross-domain figure regenerates from a single
  Python command per `REPRODUCING.md`; the action-space update is
  noted in the figure caption.

### C5b. Transfer paper updates

Land **after C3 + C4**. Closes Phase C.

- Files: `paper/main.tex` — re-render Fig 5 (dynamics adaptation,
  3-panel) and Fig 7 (cross-domain). Update §6 prose for the new
  reward + the new OfficeMedium action space; add the same
  seed-count and configuration notes as in C5a.
- Acceptance: PDF compiles; figures are produced by the
  `REPRODUCING.md` commands as committed; user has reviewed and
  approved the diff.

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

### ~~D12. Define and label the API contract test tier (D-API)~~ ✓ partial (commit `3017b5c`, 2026-05-26)

`api_contract` marker registered in `pyproject.toml`; auto-applied
by `tests/conftest.py::pytest_collection_modifyitems` to the eight
glob-listed files (`test_api.py`, `test_api_mode_default.py`,
`test_new_api.py`, `test_gym_registration.py`, `test_climate_zones.py`,
`test_data_registry.py`, `test_selection_and_env_creation.py`,
`test_types.py`). Each file received the 3-line contract header.
`pytest -m api_contract` selects exactly those eight files.

**Deferred to Phase T:** the contract-coverage acceptance
("a deliberate breaking change to `building2building/__init__.py`
exports causes at least one `api_contract` test to fail") is met
for a subset of `__all__` but not all of it. `test_new_api.py`
only `hasattr`-checks `list_building_types`, `list_buildings`,
`new_make_env`, `compute_normalized_score`, `benchmarks`,
`NormalizedDeadbandRewardConfig`, and `RewardConfig`; removing
`wrap_env_for_rl`, `Morphology`, equipment types, etc. from
`__all__` would still pass. Tightening the assertion to iterate
over the full `__all__` list belongs with the Phase T `tests/`
sweep (T27, contract-focused docstring + assertion pass).

### ~~D13. Add `CHANGELOG.md` + deprecation policy (D-API)~~ ✓ (commit `333f46b`, 2026-05-26)

`CHANGELOG.md` at repo root in Keep-a-Changelog format with an
`[Unreleased]` section and a planned `[0.1.0]` entry.
`docs/api/stability.md` lists every symbol in
`building2building/__init__.py.__all__` and documents the
deprecation policy (deprecation window, how to raise
`DeprecationWarning`, breaking-change checklist). Both are
cross-linked from the `README.md` License section and registered
in `mkdocs.yml`.

### ~~D14. Document the four benchmark problems on the docs site (D-API)~~ ✓ (commit `1e60d3c`, 2026-05-26)

One page per benchmark under `docs/benchmarks/`. The pre-existing
kebab-case filenames (`cross-domain.md`, `dynamics-adaptation.md`,
`goal-adaptation.md`, `action-transfer.md`) were extended in place
rather than creating the snake_case names in the spec; the
`mkdocs.yml` nav already pointed at the kebab-case files.
Each page now includes a metric / axis table and a minimal
`new_make_env` + benchmark-class code block, cross-linked from
`README.md`.

### ~~D1. Add MIT LICENSE~~ ✓ (commit `83711d0`, 2026-05-26)

`LICENSE` (MIT) at repo root, `pyproject.toml` `license` field, and
`README.md` footer all updated.

### ~~D2. Delete the legacy reward family~~ ✓ (commit `574baaa`, 2026-05-25)

Legacy `task1`–`task5` presets, `BarrierRewardConfig`,
`BaseRewardConfig`, and the un-normalized `DeadbandRewardConfig`
removed; only `NormalizedDeadbandReward` and the shared
`_deadband_components` helper remain. Original "D2 precedes C1"
hard dependency is replaced by "**M precedes C1**" (Phase C
preamble).

### ~~D3. Fix the documented eval bugs~~ ✓ (commit `de258ae`)

`eval_ppo.py` argument order + model path + `reward_mean` column,
`eval_dynamics_adaptation.py` argument order + metadata-driven
`PadObservation`. Regression tests under `tests/quick/`.

### ~~D4. Drop `baselines/requirements.txt`; update install docs~~ ✓ (commit `f6ec020`)

Install path is now `pip install -e ".[training]"`.

### ~~D5. Write `REPRODUCING.md`~~ ✓ (commit `45799f3`)

Lives at the repo root; cross-linked from `README.md` and
`design_doc.md`. Phase-C deliverables, Phase-B (cancelled)
calibration entries, Optuna RBC tuning, PPO CHS sweep,
smoke-test tier all documented. Will be incrementally updated by
each TODO item that adds, moves, or removes a runnable script
(Cross-phase principle 2). Outstanding follow-up: the side-by-
side PPO + SAC specialist figure command (tracked under C2).

### ~~D6. Add `baselines/` smoke tests~~ ✓ partial (commit `89d7f34`, 2026-05-26)

Landed: `tests/quick/test_train_ppo_smoke.py` (5 PPO build tests +
import smoke) and `tests/quick/test_run_reactive_control_smoke.py`
(import, `RunResult` schema, CSV schema, `_select_policy`
signature). Both run in < 5 s combined.

**Deferred to Phase T:**

- Extending the existing `tests/quick/test_train_sac_smoke.py` was
  not done. The existing file already asserts most of what D6
  asked for (`build_sac` construction, hyperparameter overrides,
  activation-fn string conversion, `PAPER_SAC_DEFAULTS` keys,
  module importability); any additional coverage is folded into
  Phase T's wrapper / training-helper rewrites.
- `tests/quick/test_eval_ppo.py` was not created. D3's regression
  contracts live in `tests/quick/test_eval_bugs.py`, which T1 and
  T27 step 3 explicitly re-scope: the worth-keeping path-layout
  contract migrates to `tests/quick/test_eval_path_layout.py`,
  and the historical CSV-column / `pad_obs_size` checks are
  dropped. No new `test_eval_ppo.py` file is needed.

### ~~D6.5. Audit and triage the test suite (precedes D7)~~ ✓ partial (commit `e1c7a3c`, 2026-05-26)

Quick-tier collection errors and stale preset names cleared in
`tests/quick/`. Two scopes were **not** addressed and are deferred
to Phase T (see new item **T-pre** below):

1. The same `BaseRewardConfig` / `DeadbandReward` import errors
   exist in `tests/long/` (`test_observation_dimension.py`,
   `test_occupancy_observation.py`, `test_random_schedule_rollout.py`,
   `test_rescale_action.py`, `test_seasonal_unoccupied.py`), causing
   `pytest -m quick` itself to abort at collection time on a fresh
   checkout because pytest collects then filters. D6.5 only swept
   `tests/quick/`.
2. Six tests in `tests/quick/test_rl_wrappers.py`
   (`test_wrap_env_for_rl_*`, `test_get_reward_params_walks_through_wrappers`,
   `test_make_rl_env_fn_returns_monitor_wrapped_env`) pass in
   isolation but fail when the full quick suite runs first — a
   collection-order leak from another test file. D6.5 noted this
   in its commit body and deferred it; it now blocks D7's CI.

Original brainstorm text retained below for the historical record.

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

### ~~D7. Add CI~~ ✓ workflow committed (commit `3017b5c`, 2026-05-26) — **red on `main` until Phase T T-pre lands**

`.github/workflows/test.yml` runs `pytest -m quick -q` on push and
PR, matrix over Python 3.10 + 3.11, after `pip install -e ".[test]"`.
The workflow file itself is correct.

The workflow currently **fails on every push** because of the two
issues catalogued under D6.5 above (long-tier `BaseRewardConfig`
import errors abort collection; six `test_rl_wrappers.py` tests
fail under the full quick-suite order). Both are deferred to Phase
T item **T-pre**; CI goes green once that lands. No production-code
change is needed in D7 itself.

### D8. Update `README.md` "Known Issues" section

After D3, D6, and D7 have landed (all done as of 2026-05-26).

- Files: `README.md`.
- Acceptance: items fixed in D3 are removed; the stale
  "No tests exist for `baselines/` code" bullet (line 203) is
  removed or narrowed now that D6 added PPO + reactive-control
  smoke tests; remaining honest limitations (legacy SAC stability
  if Phase B is incomplete at release time, CI red until Phase T
  T-pre) are listed.

### ~~D9. Documentation pass for the new reward family~~ ✓ (commit `f7fa693`, 2026-05-26)

D-lite option taken: each `docs/tutorials/*.md` was converted to
thin prose + a link to its runnable `tutorials/*.py` counterpart.
`rg 'task[1-5]'` returns nothing under `docs/` or `tutorials/`. No
code block in `docs/tutorials/*.md` duplicates code that also
lives in `tutorials/*.py`. The preset-rename open question was
already settled in D2 (`task_<mode>_<level>`); no further renaming
was needed here. Original spec kept below for reference.

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

### ~~D10. Final cleanup~~ ✓ partial (commit `1abfc42`, 2026-05-26)

`black .` pass landed across `building2building/`, `baselines/`,
`tests/`, `tutorials/`. `pyright building2building baselines` was
run; the remaining 242 errors (matplotlib/torch private-import
stubs, Hydra-typed `str → BuildingType` Literals, `PolicyLike`
gaps, pipeline internals) are catalogued in
`notes.md § "D10 — pyright status (2026-05-26)"` and left for
follow-up phases.

**Deferred:** The commit subject claims "remove dead
ThermostatSetpoint stub" but the actual diff for
`building2building/pipeline/steps/thermostat_setpoints.py` only
contains black reformatting — no removal landed.
`AddSetpointControl` / `add_setpoint_control` /
`get_temperature_setpoints` are still imported and re-exported
in `building2building/pipeline/__init__.py.__all__` despite
having no remaining call site in `building2building/` or
`baselines/`. The cleanup is folded into Phase T item **T27 step
7** (stale fixture / dead public-surface audit), which is the
natural home for "this is re-exported but called from nowhere"
removals.

### ~~D11. Delete `summaries/`~~ ✓ (commit `d0767a2`, 2026-05-26)

`summaries/` directory removed (was already untracked /
gitignored; its content had been absorbed into `notes.md`). Live
references in `notes.md` and `design_doc.md` were updated.
Remaining mentions are intentional historical context
(`notes.md:11`, `notes.md:892`) and the gitignore tripwire
(`.gitignore:54`); they do not refer to live content.

### D15. Execute the `analysis/` → `baselines/` migration

**Depends on F2's migration audit table** in
`notes.md` § "Phase F analysis/ migration audit". Each row of that
table becomes a one-commit "FX" sub-item here; this D15 entry is
the umbrella tracker for them.

For each row labelled **(promote to baselines/)**:

1. Move the script (or its useful subset) into `baselines/` at the
   destination path pinned by F2.
2. Move the plotting code into `baselines/plotting/`.
3. Rewrite imports against the public `building2building` API
   (no `from building2building.simulator.<internal>`).
4. Add a Hydra config under `baselines/configs/experiment/` if
   the script wants Hydra-style invocation.
5. Update `REPRODUCING.md` in the same commit to cite the new
   Python entry point.
6. Delete the original `analysis/` file (or leave a one-line
   pointer comment to the new location if F2's table flagged the
   audit-trail value).

For each row labelled **(delete)**: `git rm` the file in a single
sweep commit at the end.

For each row labelled **(stay in analysis/)**: no action — the
file already has a destination.

- Files: per F2's migration table; this item lists no files
  itself.
- Acceptance: `rg "from analysis" baselines/ building2building/`
  returns nothing; every script cited in `REPRODUCING.md` lives
  under `baselines/` or `building2building/`; the migration table
  in `notes.md` has every row marked done.

### D16. `REPRODUCING.md` parity sweep

**Land last in D-house, after D15.** A one-shot read-through of
every section in `REPRODUCING.md` against the actual `baselines/`
state. Catches stale Hydra group names, stale flag names, stale
output paths, and the migration's downstream effects on the
cheat-sheet table.

Per Cross-phase principle 2 this is also the "audit your
predecessors' contributions" gate — every D15 commit was
*supposed* to update `REPRODUCING.md`, but a sweep at the end
guarantees the file is internally consistent.

- Files: `REPRODUCING.md` only (no code changes here; if a
  command is broken, that's a separate bug fix).
- Acceptance: every `python -m baselines.…` command in the file
  can be at least dry-run-parsed by Hydra
  (`python -m baselines.<entry> --help` returns 0 for each entry
  point cited); the cheat-sheet table at the bottom is in 1:1
  correspondence with the in-body sections.

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

### T-pre. Unblock CI (deferred from Phase D D6.5 / D7)

**Land first in Phase T.** D7 shipped a GitHub Actions workflow
that runs `pytest -m quick -q`, but two pre-existing issues —
both partially addressed in D6.5 and explicitly deferred — keep
CI red on every push:

1. **`tests/long/` import errors.** Five files
   (`test_observation_dimension.py`, `test_occupancy_observation.py`,
   `test_random_schedule_rollout.py`, `test_rescale_action.py`,
   `test_seasonal_unoccupied.py`) still
   `from building2building.types import BaseRewardConfig`, a
   symbol deleted in D2. `pytest -m quick` collects then filters,
   so the collection errors abort the run before any test executes.
   Fix: same sweep D6.5 ran against `tests/quick/` —
   replace deleted-symbol imports with `NormalizedDeadbandRewardConfig`
   / `RewardConfig` and adapt the construction call sites. Some of
   these files are slated for relocation or deletion under T3 /
   T27 anyway; the unblocking patch only needs to keep collection
   green, not preserve every assertion.
2. **`tests/quick/test_rl_wrappers.py` ordering leak.** Six tests
   (`test_wrap_env_for_rl_observation_space_is_unit_interval`,
   `test_wrap_env_for_rl_action_in_minus_one_one`,
   `test_wrap_env_for_rl_action_default_off`,
   `test_wrap_env_for_rl_flags_are_independent`,
   `test_get_reward_params_walks_through_wrappers`,
   `test_make_rl_env_fn_returns_monitor_wrapped_env`) pass when
   the file runs in isolation but fail when the full quick suite
   runs first. A leak from an earlier test is mutating shared
   state (likely a gymnasium registry side-effect, env-var,
   working directory, or singleton in
   `building2building.simulator`). Find the offender via
   `pytest tests/quick -q --maxfail=1 -x` bisection. Fix at the
   source — either reset state in a fixture (`autouse=True`,
   `monkeypatch`-style), or rewrite the leaking test to not
   mutate shared state. **Do not paper over with
   `pytest-randomly`-style ordering pins**; the leak is real and
   blocks T24's wrapper rewrites.

The fix should land as two commits (one per root cause) so each
is bisectable. After T-pre, `pytest -m quick` exits 0 from a
fresh checkout with `pip install -e ".[test]"` and the D7 CI
workflow turns green.

- Files: `tests/long/test_*.py` (5 files for import sweep);
  whichever file is leaking state for the ordering fix.
- Acceptance: `pytest -m quick` green on a fresh checkout under
  Python 3.10 and 3.11; the GitHub Actions matrix from D7 passes.

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
   **Also covers the D10 deferred ThermostatSetpoint stub:** the
   `building2building/pipeline/steps/thermostat_setpoints.py` module
   (`AddSetpointControl`, `add_setpoint_control`,
   `get_temperature_setpoints`) is imported and re-exported from
   `building2building/pipeline/__init__.py.__all__` but has no
   remaining call site. Confirm with `rg` and remove the file +
   the `__all__` entries in the same commit. If `rg` finds a live
   call site that was missed in the D10 audit, leave the module
   alone and document the call site.
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

## Phase F — File and documentation audit (read-only)

**Run between Phase T and Phase R.** Output is a checklist, not
code changes. The phase exists because once Phase D-house and
Phase T close, the repo is *almost* OSS-ready; this is the last
chance to catch dead code, stale comments, half-migrated docs,
private paths leaking into public files, and inconsistent
narration across the docs site before paper-figure runs lock the
state of the codebase.

**Rule:** F items only **read** and **note**. Do not edit. Edits
land in a follow-up D-house or D9 commit (or in a fresh "FX"
fix-up phase if the audit surfaces more than the existing items
can absorb).

### F1. Walk every file in `building2building/`

Read every `.py`, `.yaml`, `.json`, `.md`, `.txt` under
`building2building/` and capture, per file:

1. Is the file still used? (`rg <filename> baselines/ tests/
   tutorials/ docs/` — zero hits ⇒ candidate for deletion.)
2. Is the docstring honest? (Does what it says it does, no
   stale claims about removed features.)
3. Are there `TODO` / `XXX` / `WIP` / `FIXME` comments? List them.
4. Any hardcoded paths, magic numbers, or commented-out code?
5. Any imports from `analysis/` or `baselines/` (which would
   break the public/research barrier)?
6. Is the file in the right module? (Pipeline code shouldn't
   sit under `api/`; API code shouldn't sit under `pipeline/`,
   etc.)

- Output: `notes.md` § "Phase F file audit — building2building/"
  — one bullet per file, with the six checks above as sub-bullets
  where applicable. ~1 hour read-through; the output is a flat
  list, not prose.
- Acceptance: the section is committed; every file under
  `building2building/` is mentioned at least once (even if the
  mention is "fine; no notes").

### F2. Walk every file in `baselines/` + `analysis/` migration audit

Same checks as F1, applied to `baselines/`. Extra checks specific
to `baselines/`:

7. Does the script use only the public `building2building` API?
   (`rg "from building2building" baselines/` — every import should
   resolve to a public symbol per `design_doc.md` §3.2.)
8. Are Hydra configs (`baselines/configs/`) consistent in style?
   (One naming convention, one set of default values, no
   `task_const_e0_legacy` leftovers.)
9. Are the eval scripts (`eval_ppo.py`,
   `eval_dynamics_adaptation.py`, etc.) consistent with the eval
   schema described in `REPRODUCING.md`?
10. **Is there any Slurm-only entry point** with no Python
    sibling? Per Cross-phase principle 3, every `baselines/scripts/
    *.sh` must wrap a `python -m baselines.…` invocation that
    works on a single machine. Flag any `.sh` whose body
    contains logic beyond a simple shell loop / array dispatch
    over a Python entry point.

**Plus a separate `analysis/` → `baselines/` migration audit:**

Walk every `.py` under `analysis/` and label each file as one of:

- **(stay in analysis/)** — pure exploration / decision-making
  scaffolding whose outputs are not cited in the paper. Examples:
  `analysis/officemedium_tuned/`, `analysis/_sfh_0014_sweep/`,
  ad-hoc `plot_*.py` files used to make one figure that informed
  a decision recorded in `notes.md`.
- **(promote to baselines/)** — script whose outputs **are**
  cited in the paper (including the appendix), or are part of a
  paper-figure pipeline. Examples: the SAC reward-distribution
  script that Phase R productionizes; the per-building reward-
  decomposition plotters whose figures will land in the Phase R
  appendix.
- **(delete)** — dead scripts whose outputs are not in the paper
  and whose decisions have already been baked into a YAML / a
  TODO item / a `notes.md` entry. The originating decision should
  still be reachable; the script itself can go.

For each (promote) file, name the destination file under
`baselines/` (or `baselines/plotting/` for plotting code), the
config-group rewiring needed, and any `analysis.*` imports that
need rewriting against the `building2building` public API.

- Output: `notes.md` § "Phase F file audit — baselines/" (the
  first 10 checks) and `notes.md` § "Phase F analysis/ migration
  audit" (the migration audit, as a three-column table:
  *file → destination → notes*).
- Acceptance: F1-style file roll-up plus the migration table; the
  migration table feeds **D-house FX** items that actually
  perform the moves before Phase C starts.

### F3. Walk every page in `docs/`

Same checks as F1, applied to `docs/`. Extra docs-specific
checks:

10. Does `mkdocs build --strict` pass before AND after the F3
    read-through? (Run it once to baseline, then no F3 edits
    can break it.)
11. Are there pages that duplicate content already in
    `README.md`, `REPRODUCING.md`, or `notes.md`? Flag
    duplication; D9 absorbs the de-dup work.
12. Cross-link audit: every `building2building.<symbol>`
    mention in `docs/` should hyperlink to the API page; every
    paper-figure mention should hyperlink to `REPRODUCING.md`.
13. Are there pages that still reference legacy `task1`–`task5`
    presets? (D9 is supposed to migrate these; F3 verifies.)

- Output: `notes.md` § "Phase F file audit — docs/".
- Acceptance: same shape as F1; `mkdocs build --strict` still
  passes.

### F4. Tutorials cross-check

Tutorials are 1:1 paired (`tutorials/*.py` ↔
`docs/tutorials/*.md`, per the D9 description). For each pair,
confirm they are still in sync after D9 lands (or note that they
diverged again).

- Output: `notes.md` § "Phase F tutorial sync check".
- Acceptance: every pair is checked; divergences are listed with
  a one-line note about which side is the source of truth.

### F5. Roll-up + sign-off

Merge F1–F4 into a single follow-up checklist under
`notes.md` § "Phase F roll-up". Each entry from F1–F4 that is
*not* a no-op gets mapped to either:

- (a) an existing TODO item that already covers it (cite by id),
- (b) a new "FX" follow-up item (file under "Deferred" if
  out-of-scope for the camera-ready, or under D-house if it must
  ship), or
- (c) explicitly **noted-and-accepted** (the imperfection is
  documented but won't be fixed).

- Acceptance: every non-no-op note from F1–F4 has a destination
  in (a)/(b)/(c); user signs off on the checklist before Phase R
  starts.

---

## Phase R — Empirical reward-coefficient study

**Run between Phase F and Phase C.** Output: final values of
`emed` and `ehigh` (and confirmation that `e0 = 0.0`), backed by
empirical evidence on a policy ladder and demonstrated invariance
across the `test_small` building subset.

The user-confirmed methodology (2026-05-25): reuse
`analysis/task_study/reward_design_comparison_sac.py` to train a
SAC agent while monitoring the per-term reward distribution over
training. The SAC trajectory itself provides the random →
optimal policy ladder (early training is near-random, late
training is the SAC optimum for that building).

### R0. Pin the empirical question

**Before any rollout.** Write a short methodology note
(`notes.md` § "Phase R methodology") that answers:

1. **Target invariance criterion.** What quantitative claim
   does R make? Recommendation:
   - For `emed`: at the SAC final checkpoint, the ratio
     `mean(temp_penalty / tau_T) / mean(emed * power_penalty /
     tau_E)` is in `[0.5, 2.0]` on every `test_small` building,
     i.e. the two reward terms are within a factor of 2 of each
     other end-to-end.
   - For `ehigh`: at the SAC final checkpoint, the same ratio
     drops below `0.5` (energy term dominates) on every
     `test_small` building.
   - For `e0`: trivially satisfied (`power_penalty` term has
     zero weight by construction); R0 only verifies that the
     temperature term behaves sensibly.
2. **Building subset.** `test_small` per building type. For
   compute reasons, R can restrict to a sub-subset (e.g. 2
   buildings per type, 12 total) for the *value search* and
   then validate on the full `test_small` for the chosen
   values. Pin the split in R0.
3. **Run period.** `full_year` (matches Phase A calibration).
   `winter` is faster but the calibration regime is `full_year`,
   so use `full_year`.
4. **SAC config.** Default from `baselines/configs/policy/sac.yaml`
   + `baselines/configs/training/sac.yaml`. No tuning per the
   2026-05-25 brainstorm.
5. **Coefficient grid to test.** Start with the current
   `{emed=1.0, ehigh=5.0}` plus a small grid around each:
   - `emed ∈ {0.5, 1.0, 2.0}`
   - `ehigh ∈ {3.0, 5.0, 10.0}`
   Total cells: `3 × len(building_subset)` for each coefficient
   (since the SAC trajectory itself spans the policy ladder, only
   one run per cell is needed).

- Files: `notes.md` § "Phase R methodology".
- Acceptance: every numbered choice above is pinned with a
  concrete value before any Slurm submission.

### R1. Productionize the SAC reward-distribution script into `baselines/`

`analysis/task_study/reward_design_comparison_sac.py` already
trains a SAC agent and decomposes the per-step reward into
`(temp_penalty, power_penalty)`. Since the **outputs of R3 land
in the paper appendix**, per Cross-phase principle 1 the final
script must live under `baselines/`, not `analysis/`. R1 is the
productionization commit.

Concretely:

1. Promote the script to **`baselines/run_reward_coefficient_study.py`**
   (or a similar name; pin in R1's first commit). Move only the
   logic that produces the appendix data; leave any prototype-
   only branches behind in `analysis/` (they remain there for the
   audit trail).
2. Parametrize via Hydra (matching the rest of `baselines/`):
   the energy-weight grid, the building subset, the run period,
   and the SAC config. The Hydra entry point becomes the
   canonical Python invocation cited in `REPRODUCING.md`.
3. Extend the building loop to iterate the R0-pinned subset (a
   sub-subset of `test_small`).

- Files: new
  `baselines/run_reward_coefficient_study.py`; new Hydra config
  `baselines/configs/experiment/reward_coefficient_study.yaml`;
  the prototype under
  `analysis/task_study/reward_design_comparison_sac.py` is
  preserved as historical artefact (D-house FX may later move it
  alongside the rest of the migration). Update `REPRODUCING.md`
  in the same commit (new "Appendix: reward-coefficient study"
  section under the Phase B/A block, with the Python
  invocation).
- Acceptance: a dry-run with one building × one energy weight
  produces the expected NPZ artefact under
  `outputs/reward_coefficient_study/`; the file size, schema,
  and column names match the prototype's output (so R3's
  plotting code can read it).

### R2. Run the sweep

The user runs the sweep from the Python entry point committed in
R1. The canonical invocation in `REPRODUCING.md` is the
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

### R3. Aggregate + plot

The plotting code is part of the OSS release per Cross-phase
principle 1 — it produces the paper appendix figures. Land it
under `baselines/plotting/`, not `analysis/`.

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
   ratio sits in the R0-pinned window on every building.

- Files: new
  `baselines/plotting/plot_reward_coefficient_study.py`;
  appendix-quality figures committed under `figures/appendix/`
  (or whatever the paper's existing appendix figure directory is).
  Update `REPRODUCING.md` with the plotting command alongside
  R1's training command.
- Acceptance: the three figures regenerate from one Python
  command; the values pinned in R0 are visibly satisfied (or, if
  not, R3 recommends a different pair); the figures are at
  paper-appendix quality (font size, axis labels, units).

### R4. Pin final values + update presets

Based on R3, commit the final `emed` and `ehigh` values to
`building2building/config/tasks.py` (in `TASK_PRESETS`'s
`NormalizedDeadbandRewardConfig.energy_weight` fields for the
`task_{const,occ,rand}_{emed,ehigh}` presets). Update the
README task-preset table, the docstrings, and the paper
appendix that cites these values.

- Files: `building2building/config/tasks.py`, `README.md` (task
  table values), `docs/api/types.md` (if it cites the values),
  `paper/main.tex` appendix (the chosen values + a back-reference
  to the R3 figure). The main-text task-table caption is captured
  under C5a.
- Acceptance: the values in the code match the R3 conclusion;
  `pytest -m quick tests/quick/test_task_presets.py` passes; the
  values are cited (with a forward-reference to the R3 figure)
  in the paper appendix; `REPRODUCING.md` § "Appendix: reward-
  coefficient study" is updated to lock the values.

---

## Landing the `feature/reward-normalization` branch

The original "split the 5-commit branch into 4 PRs" plan
(L0a / L0b / L1 / L2 / L3 / L4) is **superseded** as of
2026-05-25. Since that plan was written, the branch has absorbed
D2, D3, D4, D5, B1, the RL-wrapper refactor, the SAC baseline,
and the Phase-T draft. It is no longer a tidy 4-PR stack.

**Current intent (revisit before pushing):** land the branch as
one or two large PRs against `dev` once Phase M / D-house / T / F
close — not as a multi-PR stack. Phase D16's `REPRODUCING.md`
parity sweep is the natural last commit before opening the PR.

- Acceptance (revised): one or two merge commits on `dev`; the
  resulting `dev` tip can rebuild every artefact cited in
  `REPRODUCING.md` via its Python entry points.

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
