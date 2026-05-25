## Test Suite

The Building2Building test suite is intentionally lightweight. This is a
research codebase: the priority is that contributors can move quickly, not
that every code path is covered by a unit test. Tests exist to give a
**minimum level of stability** for the parts of the project that are hardest
to debug when they break silently:

- the **building processing pipeline** (`building2building/pipeline/`),
- the **public API** (`building2building.new_make_env`, `b2b.rollout`,
  `b2b.benchmarks`, the typed configs in `building2building/types.py`),
- and the **data registry** that maps building IDs from the HuggingFace
  dataset to on-disk files.

If you change those subsystems you should expect to add or update tests; for
research scripts under `analysis/` and exploratory code, tests are optional.

### Layout

```text
tests/
├── conftest.py                  # shared fixtures + EnergyPlus path setup
├── fixtures/                    # static test data (epJSON, EDD, fake dataset)
├── quick/                       # fast tests — no EnergyPlus, run in CI
└── long/                        # slow tests — require EnergyPlus + dataset
```

Three legacy test files also live at the top level (`tests/test_*.py`); see
[Legacy top-level tests](#legacy-top-level-tests) below — they predate the
`quick/` vs. `long/` split and should be migrated over time.

### Markers and selection

Two pytest markers are declared in `pyproject.toml` and applied via
`conftest.py`:

| Marker  | Meaning                                                  |
| ------- | -------------------------------------------------------- |
| `quick` | No EnergyPlus, no network, runs in seconds. Default.     |
| `long`  | Spawns real EnergyPlus simulations and/or hits HuggingFace. |

`conftest.py` auto-tags any uncategorised test as `quick`, so unmarked files
behave conservatively. Most long tests additionally gate on the environment
variable `B2B_RUN_LONG_TESTS=1` so that an accidental `pytest tests/long`
invocation does not blow through a CPU budget.

Common invocations:

```bash
pytest -m quick                                # default CI-style run
B2B_RUN_LONG_TESTS=1 pytest -m long            # full long suite (slow!)
pytest tests/quick/test_task_presets.py        # one file
pytest -k "deadband and not legacy"            # keyword filter
```

### Shared fixtures

`tests/conftest.py` exposes three fixtures backed by `tests/fixtures/`:

| Fixture            | What it provides                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| `fake_dataset_dir` | Path to a minimal stand-in for the HuggingFace dataset (one parquet + one `splits.json`).                  |
| `fake_metadata`    | The fake metadata parquet loaded as a `pandas.DataFrame`.                                                  |
| `fake_splits`      | The fake `splits.json` loaded as a nested `dict`.                                                          |
| `baseline_csv_path`| Path to a small `baseline_returns_fixture.csv` used by scoring tests.                                      |

In addition, `conftest.py` calls `building2building.env.setup_energyplus_path()`
once at collection time (guarded by `ModuleNotFoundError`), so any test that
imports `building2building` gets a usable EnergyPlus install if one is
configured locally.

---

## Quick tests (`tests/quick/`)

These tests must not require EnergyPlus, must not hit the network, and
should each complete in under a second. They are the safety net you run
before pushing.

### Public API surface

- **`test_new_api.py`** — Asserts that the top-level `building2building`
  package re-exports the headline symbols (`list_building_types`,
  `list_buildings`, `new_make_env`, `compute_normalized_score`, `benchmarks`,
  and the reward config dataclasses). A change that breaks one of these
  imports is almost always a regression.
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

### Wrappers, scoring, training scaffolding

- **`test_rl_wrappers.py`** — Smoke tests for the public RL wrappers.
  Checks that `wrap_env_for_rl(normalize_obs=True)` advertises a
  `[0, 1]` observation space, that `new_make_env(rescale_action=True)`
  advertises a `[-1, 1]` action space, that the two flags are independent,
  and that `make_rl_env_fn` returns a `Monitor`-wrapped env. **Note:** these
  use `new_make_env` against a real building, so they currently require the
  dataset and EnergyPlus to be available — they are functionally `long`
  tests even though they live in `quick/`.
- **`test_rollout.py`** — Quick part: round-trips a fake `Trajectory`
  through `.to_npz` / `.from_npz`, including `infos`, `raw_observations`,
  `controlled_zones`, and `observation_names`; verifies
  `callable_controller` and the public re-exports. The long part
  (`TestRolloutEndToEnd`) runs a 5-step rollout against a real
  `OfficeSmall` env.
- **`test_scoring.py`** — Tests `compute_normalized_score` with a hand-built
  cache: per-building lookups, missing-key errors, the zero-baseline
  fallback, and the requirement that `building_id` be provided.
- **`test_benchmarks.py`** — Exercises every benchmark class:
  `GoalAdaptation`, `DynamicsAdaptation` (with `easy`/`medium`/`hard`),
  `ActionSpaceTransfer` (unitary vs. central, expand vs. reduce, the
  `_unitary_sat_overrides` and `_central_sat_overrides` helpers, the
  `additional_fixed` plumbing on `hvac_action_space`, and the
  `_compute_overrides` direction logic), and `CrossDomainGeneralization`.
- **`test_eval_bugs.py`** — Regression tests for the D3 eval-script bug
  batch: nested model-path parsing in `baselines/eval_ppo`, the
  `reward_mean` column rename, `pad_obs_size` becoming a keyword-only
  parameter on `evaluate_multi_building`, and the train/eval
  `metadata.json` round-trip.
- **`test_train_sac_smoke.py`** — Smoke tests for the SAC harness in
  `baselines/utils/training`: `build_sac` constructs on `Pendulum-v1`,
  accepts hyperparameter overrides, and parses the `activation_fn` string
  into a `torch.nn` class. Also asserts the keys and value types of
  `PAPER_SAC_DEFAULTS` and that the SAC training modules import without
  circular dependencies.

---

## Long tests (`tests/long/`)

These tests spawn real EnergyPlus simulations or download the published
dataset. Each one short-circuits via `B2B_RUN_LONG_TESTS=1`, so they only
run when the user explicitly opts in:

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

### Pipeline / single-zone houses

- **`test_pipeline_single_zone_houses.py`** — End-to-end check that a
  `SingleFamilyHouse` env can be built via `new_make_env` and stepped.
  This is the only test that exercises the SFH-specific code paths
  (`Schedule:File` rewriting, no climate-zone bucket).

### Observation-space contracts

- **`test_observation_dimension.py`** — Asserts that the observation
  dimension changes with `target_temperature_mode`: `constant` mode drops
  the occupancy and target-temperature slots, `occupancy` mode adds two
  slots per controlled zone, and `random_schedule` (`task_rand_*`) matches
  `occupancy` exactly.
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

### Action-space wrappers

- **`test_rescale_action.py`** — Wraps `OfficeSmall` and
  `SingleFamilyHouse` envs in `gym.wrappers.RescaleAction(-1, 1)` and
  asserts that the bounds change, the action dimensions don't, and the
  env still steps successfully when fed samples from the wrapped action
  space.

---

## Legacy top-level tests

Three files live directly in `tests/` (outside `quick/` and `long/`).
They predate the marker split and should be moved when convenient:

- **`test_building_param_clipping.py`** — Single test that confirms
  `AugmentObservationWithBuildingParams` clips normalized building
  parameters into `[-1, 1]` even when the raw values are far out of
  range. Uses a `MagicMock` env, so it is effectively `quick`.
- **`test_get_actuators.py`** — Two tests for the EDD parser
  `get_hvac_actuators`. The first runs against the committed
  `tests/fixtures/eplusout.edd` (quick). The second runs EnergyPlus on
  the bundled `bldg1.epjson` to regenerate the EDD file
  (`long`-marked).
- **`test_wrappers.py`** — Unit tests for the three observation wrappers
  `AugmentObservationWithBuildingParams`, `NormalizeObservation`,
  and `PadObservation`, all driven by a hand-written `MockEnv`. The
  zone-aware padding test is the most useful one: it documents that the
  last seven observation slots (outdoor temperature, outdoor humidity,
  time of day, day of week, day of year, electricity, gas) must remain
  at consistent indices across buildings with different numbers of
  zones — this invariant is what makes the multi-building generalization
  benchmarks work.

---

## What is intentionally **not** covered

The following key features are currently *not* exercised by tests; add
coverage with care before relying on them in publications:

- **End-to-end pipeline** (`building2building/pipeline/`): the multi-step
  IDF → epJSON → controllable building pipeline (`prepare_building`,
  `create_complete_pipeline`, `make_controllable`,
  `extract_discovery_metadata`) has **no test**. Only the EDD parser and
  the very last actuator-discovery step are touched. This is the single
  largest gap relative to the priorities stated at the top of this page.
- **HuggingFace download path** (`building2building/data/download.py`) is
  mocked everywhere. There is no integration test that asserts the real
  archive layout matches what `BuildingRegistry` expects — `long`
  tests would hit it implicitly, but only in the happy path.
- **Morphology graph** (`building2building/morphology.py`) and the
  per-node observation/action decomposition advertised on the docs
  landing page have **no tests**.
- **Hydra/CLI entry points** (`baselines/train_ppo.py`,
  `baselines/eval_*.py`) are imported in `test_eval_bugs.py` and
  `test_train_sac_smoke.py` but never invoked with real configs.
- **Wrappers** beyond the three covered in `test_wrappers.py`
  (`ResampleAction`, the multi-building padding pipeline used in
  cross-domain transfer) are not unit-tested.
- **`scoring._cache` loading from CSV**: only the in-memory dispatch is
  tested; the CSV loader is never invoked with a real file.

If you extend any of these, please add at least one quick test that
documents the contract — that is what these tests are for.
