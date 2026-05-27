## Test Suite

The Building2Building test suite is intentionally lightweight. This is a
research codebase: the priority is that contributors can move quickly, not
that every code path is covered by a unit test. Tests exist to give a
**minimum level of stability** for the parts of the project that are hardest
to debug when they break silently:

- the **building processing pipeline** (`building2building/pipeline/`),
- the **public API** (`building2building.new_make_env`, `b2b.rollout`,
  `b2b.benchmarks`, the typed configs in `building2building/types.py`),
- the **observation and action wrappers** (`building2building/simulator/wrappers.py`),
- and the **data registry** that maps building IDs from the HuggingFace
  dataset to on-disk files.

If you change those subsystems you should expect to add or update tests; for
research scripts under `analysis/` and exploratory code, tests are optional.

---

## Test design principles

> **Note:** the canonical version of this section lives in `TODO.md` under
> "Phase T — Test suite cleanup and core coverage" for as long as Phase T is
> in flight. The snapshot below was taken from that block; update `TODO.md`
> first and then re-copy here.

These principles govern every test PR in this project. They exist because a
test that lives in its own parallel universe — hand-rolled mocks, fake config
dicts, `MagicMock(spec=...)` stand-ins for B2B objects — silently drifts as
the production code evolves. The real code changes shape, the mock keeps
passing, and the test becomes false reassurance.

1. **Exercise the real code path.** Build envs through real
   `new_make_env` / `make_env_from_config` against a committed
   fixture; do not build a `MockEnv` that pretends to be a B2B env.
   Construct configs (`TaskConfig`, `EnvBuildConfig`,
   `BuildingInfo`, `BuildingConfig`) via their real
   constructors / `from_dict` / `from_json` methods, not via
   `MagicMock(spec=...)` or hand-rolled dicts.
2. **Mock only at external I/O boundaries.** The HuggingFace
   network call is the only legitimate mock target — and even then,
   the preferred pattern is to point the registry at the committed
   minimal fixture via the shared `fixture_registry` helper (T0),
   not to monkeypatch ad-hoc. EnergyPlus itself is not mocked.
3. **Minimal, shared, real fixtures.** Prefer the committed
   minimal-building matrix (one fixture per HVAC type) +
   one committed `equipment.json` per HVAC type reused across
   the whole quick suite, over per-test hand-rolled fixtures.
   Every fixture is documented in a `README.md` next to it so its
   provenance is reproducible.
4. **Mock surface is itself a code smell.** If a test needs more
   than ~5 lines of `monkeypatch` / `MagicMock` setup, that is a
   signal the test is at the wrong layer. Extract a small helper
   from production code and test that directly.
5. **Wrappers and other generic `gym.Env` consumers may use a
   `MockEnv`** — those are general by design. But where the
   wrapper interacts with B2B-specific metadata (e.g.
   `PadObservation` reading `observation_names`), the test must
   *also* gain a companion test against the real minimal-building
   fixture, so a wrapper that works on `MockEnv` but breaks on the
   real `metadata` / `observation_space` shape is caught.

These are guidelines, not absolute rules — the goal is fewer
sources of false positives, not maximum integration-test purity.

---

### Layout

```text
tests/
├── conftest.py                  # shared fixtures + EnergyPlus path setup
├── fixtures/                    # static test data (epJSON, equipment.json, fake dataset)
├── quick/                       # short rollouts (≤ ~20 steps); EnergyPlus allowed
├── long/                        # multi-day rollouts; gated on B2B_RUN_LONG_TESTS=1
└── release/                     # dataset / artifact integrity against published HF data
```

### Markers and selection

Three pytest markers are declared in `pyproject.toml` and applied via
`conftest.py`:

| Marker    | Meaning                                                                                                                                  |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `quick`   | No rollout, or a short rollout (≤ ~20 steps). EnergyPlus and the cached HF dataset are allowed. Aim for each test under a few seconds.  |
| `long`    | Multi-day rollouts (hundreds to tens of thousands of steps), or per-cycle leak/lifecycle iteration. Gated on `B2B_RUN_LONG_TESTS=1`.    |
| `release` | Dataset / artifact integrity checks against the published `vtaboga/building2building_dataset`. Not run on every push; lives in `tests/release/`. CI automation is deferred item TZ1. |

`conftest.py` auto-tags any uncategorised test as `quick`, so unmarked files
behave conservatively. Most long tests additionally gate on the environment
variable `B2B_RUN_LONG_TESTS=1` so that an accidental `pytest tests/long`
invocation does not blow through a CPU budget.

Common invocations:

```bash
pytest -m quick                                # default CI-style run
B2B_RUN_LONG_TESTS=1 pytest -m long            # full long suite (slow!)
pytest -m release                              # dataset-integrity checks (needs real HF data)
pytest tests/quick/test_task_presets.py        # one file
pytest -k "deadband and not legacy"            # keyword filter
```

### Shared fixtures

`tests/conftest.py` exposes three fixtures backed by `tests/fixtures/`:

| Fixture             | What it provides                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| `fake_dataset_dir`  | Path to a minimal stand-in for the HuggingFace dataset (one parquet + one `splits.json`).               |
| `fake_metadata`     | The fake metadata parquet loaded as a `pandas.DataFrame`.                                               |
| `fake_splits`       | The fake `splits.json` loaded as a nested `dict`.                                                       |
| `baseline_csv_path` | Path to a small `baseline_returns_fixture.csv` used by scoring tests.                                   |
| `fixture_registry`  | A `BuildingRegistry` pointed at the committed minimal-building fixtures (one per HVAC type).            |

In addition, `conftest.py` calls `building2building.env.setup_energyplus_path()`
once at collection time (guarded by `ModuleNotFoundError`), so any test that
imports `building2building` gets a usable EnergyPlus install if one is
configured locally.

---

## Quick tests (`tests/quick/`)

These tests run a short rollout (≤ ~20 steps) or no rollout at all.
EnergyPlus startup (~1–3 s) is acceptable. They are the safety net you run
before pushing.

### Public API surface

- **`test_api.py`** — Smoke tests for `list_building_types` and
  `list_buildings`. Uses `unittest.mock.patch` to stub out the HuggingFace
  registry so the call delegates correctly without touching the network.
- **`test_api_mode_default.py`** — Regression guard for `new_make_env`. A
  previous bug silently defaulted `target_temperature_mode` to `"constant"`,
  which silently disabled occupancy-driven setpoints for `task_occ_*`/`task_rand_*`. The
  test monkey-patches `create_simulator` to raise a marker exception as soon
  as the `TaskConfig` has been built, then inspects the captured config to
  verify the preset's mode wins when the caller omits the override.
- **`test_gym_registration.py`** — Verifies that importing
  `building2building` registers a `b2b/<BuildingType>-v0` entry in the
  Gymnasium registry for every building type, and that
  `register_all` is idempotent.

### Typed config dataclasses

- **`test_types.py`** — Tests `RunPeriodConfig`, `ZoneTargetTemperatureConfig`,
  `TaskConfig`, `NormalizedDeadbandRewardConfig`, `RandomScheduleConfig` and
  the dispatch function `reward_config_from_dict`.
  Covers default values, `from_dict` parsing, season validation, the seasonal
  unoccupied schedule round-trip, and the `expected_steps` arithmetic for run
  periods.
- **`test_config_models.py`** — Tests `DatasetSelectionConfig` and
  `EnvBuildConfig` (the Hydra-friendly wrappers). Includes the
  building-type allow-list (no `HotelSmall`, exactly six types) and the
  frozen-dataclass guarantee.
- **`test_task_presets.py`** — Documents the nine normalized presets
  (`task_<mode>_<level>` for `mode ∈ {const, occ, rand}`,
  `level ∈ {e0, emed, ehigh}`). Each preset is
  inspected for the right reward type, energy weight, setpoint mode, and
  unoccupied policy. Two additional tests guard that:

  1. Normalized presets are stored *unfilled* (`tau_T = tau_E = None`) so
     that `new_make_env` resolves them per-building from the YAML.
  2. The `make_normalized_deadband_task` factory does its YAML import
     lazily — importing `config.tasks` must not trigger a metadata
     download.

- **`test_selection_and_env_creation.py`** — Round-trips
  `DatasetSelectionConfig`, `EnvBuildConfig`, and `parse_benchmark_config`
  through their `from_dict` constructors.

### Data registry and reward normalizers

- **`test_data_registry.py`** — Uses the `fake_dataset_dir` fixture to drive
  `BuildingRegistry` through every public method (`list_building_types`,
  `list_buildings`, `get_building_by_index`, `get_building_by_id`,
  `query_buildings`). Includes negative paths (out-of-range index, missing
  ID, empty split).
- **`test_climate_zones.py`** — Covers `BuildingRegistry.list_buildings_by_climate_zone`,
  `ClimateZoneUnavailableError`, the public re-exports
  (`b2b.list_buildings_by_climate_zone`, `b2b.get_climate_zone`), and the
  hard-fail behaviour when the parquet is missing the `climate_zone`
  column. Also has a `long`-marked sub-class that re-runs the same checks
  against the real published dataset (`vtaboga/building2building_dataset`).
- **`test_reward_normalizers.py`** — Writes synthetic
  `reward_normalizers.yaml` files to a temp dir and exercises the parser,
  the schema-version check, the per-bucket floor logic
  (`epsilon_abs`/`epsilon_rel`), and `_cz_key_for` for SFH (which has no
  climate zone). Also asserts that `DEFAULT_REWARD_NORMALIZERS_PATH` points
  to a real YAML inside the installed package.

### Rewards, schedules, observations

- **`test_normalized_deadband_reward.py`** — Tests
  `NormalizedDeadbandRewardConfig` (filled vs. unfilled invariants, error
  cases), `NormalizedDeadbandReward` (halves the temperature contribution when
  `tau_T = 2.0`), and `_maybe_warn_normalized_deadband` (only the
  calibration regime — `occupancy` mode with `dT=1` — is silent; everything
  else emits a deduplicated `RuntimeWarning`).
- **`test_random_schedule.py`** — Tests the random daily schedule generator
  used by `task_rand_*` presets: `DailySchedule` invariants, `month_to_season` mapping,
  `BUILDING_TYPE_DEFAULTS` coverage, `TruncatedNormal`, determinism given a
  seed, variation across days and building types, and the seasonal bias
  (winter: unoccupied ≤ occupied; summer: unoccupied ≥ occupied).
- **`test_seasonal_target.py`** — Tests
  `ZoneTargetTemperatureConfig.unoccupied_for_season` and the runtime
  helper `DynamicTargetTemperature` for `policy="seasonal"`. Asserts that
  `task_occ_*` ships with the seasonal map `winter=18, shoulder=21, summer=26`.

### Building pipeline

- **`test_pipeline_prepare_building.py`** — End-to-end check that
  `prepare_building` converts raw IDF fixtures through the full pipeline
  (upgrade → convert → add outputs → make controllable) for both a VAV
  multi-zone building and an SFH with `Schedule:File` rewriting.
- **`test_pipeline_make_controllable.py`** — Pins the actuator-emission
  contract for `make_controllable` per HVAC type (VAV, Unitary,
  HeatingOnly): asserts that each required `(object-type, actuator-type)`
  pair is present in the output.
- **`test_pipeline_discovery.py`** — Pins `extract_discovery_metadata`
  (net conditioned area, warmup phases, HVAC actuator count) against values
  committed in each minimal fixture's `README.md`.
- **`test_patch_epjson_run_period.py`** — Pins the `_patch_epjson_run_period`
  contract: rewrites `RunPeriod` dates and `Schedule:File` paths to the
  correct seasonal CSV; handles missing `RunPeriod` objects and the
  `"summer"` period alias.
- **`test_equipment_schema.py`** — Pins the equipment-schema round-trip
  (cattrs `structure` / `unstructure`) for each HVAC type (VAV, Unitary,
  HeatingOnly). Asserts the expected schema class and that actuator
  descriptions survive the round-trip.
- **`test_officemedium_actuator_set.py`** — Regression test for the
  OfficeMedium OA-mixer actuator-emission fix (Phase M item M1). Pins that
  `make_vav_system_controllable` emits exactly one
  `Outdoor Air Controller × Air Mass Flow Rate` actuator per air loop, and
  that no schedule or EMS program silently overrides an agent-facing actuator.

### Environment construction

- **`test_new_make_env_minimal.py`** — Pins the knobs-and-HVAC-matrix
  contract for `new_make_env`: every combination of run period,
  target-temperature mode, and HVAC type (VAV / Unitary / HeatingOnly)
  produces an env whose `reset()` returns an observation of the correct
  dtype and whose `action_space` has the expected shape. Also verifies that
  `rescale_action=True` and `max_episode_steps` are honoured.
- **`test_new_make_env_cleanup.py`** — Pins the staging-directory cleanup
  contract: seasonal run periods register a cleanup callback for the
  EnergyPlus staging directory on `env.close()`; full-year periods skip
  the cleanup (the staging dir is reused across episodes).

### Wrappers and observation layout

- **`test_wrap_env_for_rl.py`** — Pins the `wrap_env_for_rl` composition
  and rescaling contract. Asserts that `normalize_obs` and `rescale_action`
  flags are independent, that their composition order places
  `NormalizeObservation` outermost, that actions in `[-1, 1]` are correctly
  mapped to physical actuator ranges, and that `env.metadata` is preserved
  through the wrapper stack. Includes a real-env companion test.
- **`test_normalize_observation.py`** — Pins the `NormalizeObservation`
  wrapper contract: observations are mapped affinely into `[0, 1]` using
  observation-space bounds, the transformation is invertible, a zero-range
  dimension raises at construction, and `reset()` rebuilds the bounds when
  the underlying space changes. Includes a real-env companion test.
- **`test_augment_building_params.py`** — Pins the
  `AugmentObservationWithBuildingParams` contract: building parameters are
  extracted from env metadata, normalized to `[-1, 1]`, and appended to
  the observation vector. Also verifies that missing metadata raises by
  default (fail-loud), that `allow_defaults` falls back gracefully, and
  that `reset()` re-reads metadata when the underlying env changes.
  Includes a real-env companion test.
- **`test_resample_building_wrapper.py`** — Pins the
  `ResampleBuildingOnResetWrapper` contract: raises on empty index lists,
  a single-index list yields a stable env on every reset, a multi-index
  list swaps the underlying env on reset, episode counters reset correctly
  across swaps, and index-out-of-bounds errors are warned about and
  deferred to the next reset rather than crashing immediately.
- **`test_pad_observation.py`** — Pins the `PadObservation` zone-split
  and padding contract: zone features are split from non-zone features
  using `observation_names` metadata, zone-padding zeros are inserted
  between zone slots and the non-zone tail, non-zone features always appear
  at the end of the padded vector, and the wrapper rebuilds its layout
  after `reset()` when the underlying env changes shape.
- **`test_obs_padding_invariants.py`** — Companion test for `PadObservation`
  against real environments. Asserts that after padding, the seven non-zone
  tail features (`time_of_day`, `day_of_week`, `day_of_year`,
  `outdoor_temperature`, `outdoor_humidity`, `energy_gas`, `energy_electricity`)
  appear at the same fixed indices regardless of the number of zones, using
  a real env built from `fixture_registry`.
- **`test_observation_names_stability.py`** — Snapshot test that pins the
  exact list of observation names returned by the minimal-VAV fixture env
  for each target-temperature mode (`constant`, `occupancy`,
  `random_schedule`). Catches any change to the observation vector
  structure before it silently breaks trained models.

### Scoring and training scaffolding

- **`test_rollout.py`** — Quick part: round-trips a fake `Trajectory`
  through `.to_npz` / `.from_npz`, including `infos`, `raw_observations`,
  `controlled_zones`, and `observation_names`; verifies
  `callable_controller` and the public re-exports. The long part
  (`TestRolloutEndToEnd`) runs a 5-step rollout against a real
  `OfficeSmall` env.
- **`test_scoring.py`** — Tests `compute_normalized_score` with a hand-built
  cache: per-building lookups, missing-key errors, the zero-baseline
  fallback, and the requirement that `building_id` be provided.
- **`test_scoring_csv.py`** — Exercises the real CSV-loading path in
  `building2building.scoring`: loads a fixture CSV via `scoring._load()`,
  verifies the key structure, checks missing-key errors, and asserts
  that `CSV_PATH` points at a real file inside the installed package.
- **`test_benchmarks.py`** — Exercises every benchmark class:
  `GoalAdaptation`, `DynamicsAdaptation` (with `easy`/`medium`/`hard`),
  `ActionSpaceTransfer` (unitary vs. central, expand vs. reduce, the
  `_unitary_sat_overrides` and `_central_sat_overrides` helpers, the
  `additional_fixed` plumbing on `hvac_action_space`, and the
  `_compute_overrides` direction logic), and `CrossDomainGeneralization`.
- **`test_eval_path_layout.py`** — Pins the eval model-path layout
  contract: `_parse_model_path` must interpret the nested
  `<building_type>/<task>/ppo_<id>.zip` directory structure correctly, and
  `rglob` must discover those files while a flat `glob` does not.
- **`test_train_sac_smoke.py`** — Smoke tests for the SAC harness in
  `baselines/utils/training`: `build_sac` constructs on `Pendulum-v1`,
  accepts hyperparameter overrides, and parses the `activation_fn` string
  into a `torch.nn` class. Also asserts the keys and value types of
  `PAPER_SAC_DEFAULTS` and that the SAC training modules import without
  circular dependencies.
- **`test_train_ppo_smoke.py`** — Smoke tests for the PPO harness:
  `build_ppo` constructs on `CartPole-v1` without raising, and
  `PAPER_PPO_DEFAULTS` has the expected keys and value types. Also asserts
  the module imports correctly (no circular dependencies).
- **`test_run_reactive_control_smoke.py`** — Smoke tests for the
  reactive-control baseline: module imports without circular dependencies,
  `_select_policy` correctly dispatches to `AirLoopPolicy` vs.
  `UnitaryHvacPolicy`, `write_csv` produces the expected columns, and
  `RunResult` has the required fields.

---

## Long tests (`tests/long/`)

These tests run multi-day rollouts or iterate over many create/reset/close
cycles. Each one short-circuits unless `B2B_RUN_LONG_TESTS=1`:

```bash
B2B_RUN_LONG_TESTS=1 pytest tests/long -s
```

### Environment lifecycle and resource leaks

- **`test_env_lifecycle.py`** — Vanilla Gymnasium API checks against a real
  `SingleFamilyHouse / task_const_e0 / winter` env: `reset` returns
  `(np.ndarray, dict)`, `step` returns a 5-tuple, `observation_space.contains(obs)`,
  and a 10-step episode runs without crashing.
- **`test_env_leak.py`** — Acceptance criteria for TODO B0 (env leak fix):

  1. `env.close()` leaves no leftover EnergyPlus output directory.
  2. `threading.active_count()` returns to its baseline after every
     `close()` (and after every `reset()` on a persistent env).
  3. RSS growth over `N=20` create/reset/close cycles stays below a
     measured bound that accounts for the known ~14 MB/cycle of
     EnergyPlus-native residual.

  The `psutil`-based RSS check is auto-skipped if `psutil` is not
  installed.
- **`test_all_buildings_env_smoke.py`** — Parallel smoke test across **every**
  building in the unified metadata. Buildings are processed in batches of
  `BATCH_SIZE = 50` in fresh subprocesses (so the EnergyPlus memory leak
  cannot OOM-kill the test), up to `NUM_WORKERS = 3` batches concurrently,
  with `BATCH_TIMEOUT_S = 600`. Each batch flushes incremental results to a
  JSON file so a mid-batch crash does not destroy the rest of the run.

### Observation-space contracts

- **`test_occupancy_observation.py`** — Runs ~2 days of an `OfficeSmall`
  env in occupancy mode and asserts that at least one zone sees both
  occupied and unoccupied timesteps, and that the target temperature
  exactly tracks the configured `occupied_c` / `unoccupied_c` values.

### Task-specific behaviour

- **`test_seasonal_unoccupied.py`** — End-to-end rollout for `task_occ_*`'s
  seasonal unoccupied policy. Runs three windows: a summer window (mean
  unoccupied target ≥ 24 °C), a winter window (mean unoccupied target ≤
  19 °C), and a full year (summer mean strictly above winter mean by at
  least 3 °C).
- **`test_random_schedule_rollout.py`** — End-to-end rollout for `task_rand_*`'s
  random daily schedule:

  - Multiple distinct setpoints per zone over 3 days.
  - Same seed → byte-identical target traces (both via
    `make_env_from_config` and via `new_make_env`).
  - Different seeds → traces differ.
  - `zone_occupancy ∈ {0, 1}` and is consistent with the target
    transitions.

### Benchmarks and dataset generation

- **`test_benchmarks_behaviour.py`** — Behaviour smoke-tests for benchmark
  env factories using the real data registry. Resets and steps one env
  from each of `DynamicsAdaptation`, `GoalAdaptation`,
  `CrossDomainGeneralization`, and `ActionSpaceTransfer` to confirm the
  factory plumbing works end-to-end.
- **`test_generate_raw_dataset_matches_existing.py`** — Stage 1 validation
  (Phase G item G3): regenerates IDF fixtures and asserts that the resulting
  epJSON metadata rows match the upstream HuggingFace reference for
  `(building_type, place, source_idf, weather_file)` columns and LHS
  parameter values.
- **`test_generate_dataset.py`** — Stage 2 smoke test (Phase G item G4):
  regenerates one `OfficeMedium` building via `generate_dataset.py` and
  validates the written artifacts: `equipment.json` round-trips through
  cattrs, `action_dim` in the metadata parquet matches the actuator count,
  and `splits.json` is bit-identical to the HF upstream copy.

---

## Release tests (`tests/release/`)

These tests check the integrity of the published dataset and baseline
artifacts against the live `vtaboga/building2building_dataset` on
HuggingFace. They are **not** run on every push; CI automation is deferred
item TZ1. Run them manually before a release:

```bash
pytest -m release
```

- **`test_data_integrity.py`** — Asserts that every building ID in `splits.json`
  appears in `metadata.parquet`, that the `train` and `test` splits do not
  overlap per building type, and that `test_small` is a strict subset of
  `test`.
- **`test_reward_normalizers_coverage.py`** — Asserts that
  `reward_normalizers.yaml` has an entry for every
  `(building_type, climate_zone)` bucket present in the published
  metadata, and that `resolve_reward_normalizer` succeeds for every row.
- **`test_baseline_returns_coverage.py`** — Asserts that
  `baseline_returns.csv` contains a row for every
  `(building_type, task, run_period, building_id)` tuple in the paper
  evaluation grid (as declared in `eval_reactive_control.yaml`).

---

## What is intentionally **not** covered

The following key features are currently *not* exercised by tests; add
coverage with care before relying on them in publications:

- **HuggingFace download path** (`building2building/data/download.py`) is
  mocked everywhere. There is no integration test that asserts the real
  archive layout matches what `BuildingRegistry` expects — `long`
  tests would hit it implicitly, but only in the happy path.
- **Morphology graph** (`building2building/morphology.py`) and the
  per-node observation/action decomposition advertised on the docs
  landing page have **no tests**.
- **Hydra/CLI entry points** (`baselines/train_ppo.py`,
  `baselines/eval_*.py`) are imported in `test_eval_path_layout.py` and
  `test_train_sac_smoke.py` but never invoked with real configs.

If you extend any of these, please add at least one quick test that
documents the contract — that is what these tests are for.
