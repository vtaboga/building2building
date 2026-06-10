# Phase T – Final Sweep Checklist (T27a)

Audit artifact for T27. Each item records the exact files/tests/fixtures to
touch and the decision taken. **T27a does not change any test file**; it only
commits this checklist. T27b executes every decision one-to-one.

---

## Item 1 – No tests directly under `tests/` other than `conftest.py`

Two legacy files remain at the top level after T2.

### `tests/test_get_actuators.py`

Both tests (`test_get_hvac_actuators_from_edd`, `test_get_hvac_actuators_with_simulation`)
test `get_hvac_actuators()` (exported from `building2building.pipeline`).

Call-site audit:
```
grep -rn "get_hvac_actuators" building2building/ --include="*.py" \
  | grep -v "parse_edd.py\|pipeline/__init__.py"
```
Returns empty — the function is re-exported in `__all__` but has no live
production call sites.  Testing a dead export pins no meaningful contract and
creates maintenance surface for nothing.

**Decision: `git rm tests/test_get_actuators.py`** (both tests dropped).
This also unblocks deletion of `tests/fixtures/eplusout.edd`,
`tests/fixtures/bldg1.epjson`, and `tests/fixtures/weather.epw` in Item 7.

Note: `get_hvac_actuators` itself is a candidate for removal from the public
API, but that is a D-phase concern and out of scope for T27.

### `tests/test_wrappers.py`

Contains `TestAugmentObservationWithBuildingParams`, `TestNormalizeObservation`,
`TestPadObservation` — all using a local `MockEnv`.

Coverage audit:
| test class | superseded by |
|---|---|
| `TestAugmentObservationWithBuildingParams` | `tests/quick/test_augment_building_params.py` (5 tests, same invariants plus round-trip) |
| `TestNormalizeObservation` | `tests/quick/test_normalize_observation.py` (4 tests, same boundaries) |
| `TestPadObservation` | `tests/quick/test_pad_observation.py` (4 tests) + `test_obs_padding_invariants.py` (real-env) |

**Decision: delete `tests/test_wrappers.py`** — no test body is unique.

---

## Item 2 – Delete `tests/long/test_pipeline_single_zone_houses.py`

File contains one test: `test_single_family_house_env_creation`.
- `max_episode_steps=8`, calls `reset()` once and `step()` once.
- Asserts only that neither call crashes and `action_space.shape[0] > 0`.
- The docstring itself says "Audit note: temporary long marker; scheduled for
  T27 deletion."
- T5 SFH parametrize case and T20 benchmark smoke test cover the SFH path
  with actual pipeline assertions.

**Decision: `git rm tests/long/test_pipeline_single_zone_houses.py`.**

---

## Item 3 – `tests/quick/test_eval_bugs.py` re-evaluation

Current classes and decisions:

| class | decision | rationale |
|---|---|---|
| `TestParseModelPath` (5 tests) | **keep** | Pins a real path-layout contract: `_parse_model_path` must parse the nested `<type>/<task>/ppo_<id>.zip` directory structure, and `rglob` must discover it. Removing this risks a silent regression in the eval entry points. |
| `TestEvalPpoCsvRewardMeanColumn` (1 test) | **delete** | Tests the historical D3 rename from `reward` to `reward_mean`. The field is stable; the test checks `hasattr` on a dataclass — not a meaningful ongoing contract. |
| `TestEvaluateMultiBuildingSignature` (2 tests) | **delete** | Tests that `pad_obs_size` exists as a keyword-only parameter. Signature-inspection tests are brittle and add no behavioral coverage beyond what calling the function provides. |
| `TestDynamicsAdaptationMetadataRoundTrip` (1 test) | **delete** | The body is entirely synthetic (`pytest.raises(FileNotFoundError): raise FileNotFoundError(…)`); it tests no real code path. |

Survivors: `TestParseModelPath` only.

**Decision: delete the three dead classes; rename the file to
`tests/quick/test_eval_path_layout.py`.**  Update the module docstring to
reflect the surviving contract.

---

## Item 4 – `MagicMock(spec=<B2B class>)` audit

```
grep -rn "MagicMock(spec" tests/ --include="*.py"
```

Returns empty — no matches.

**Decision: no action.**

---

## Item 5 – Hand-rolled config dict audit

```
grep -rn '"target_temperature_mode"\|"reward_config"' tests/ --include="*.py"
```

All occurrences are dict literals passed directly to `TaskConfig.from_dict()`
or used as `pytest.mark.parametrize` values (not bypassing constructors).
No test constructs a B2B dataclass by passing a raw dict where a constructor
call is expected.

**Decision: no action.**

---

## Item 6 – Wrapper companion-test audit

For each T24 wrapper test file (MockEnv-based), one companion test is needed
that instantiates the same wrapper against a real env built via
`fixture_registry`.

| wrapper | existing MockEnv tests | real-env companion? | action |
|---|---|---|---|
| `AugmentObservationWithBuildingParams` | `test_augment_building_params.py` | none | **add** `test_augment_building_params_real_env` to `test_augment_building_params.py` |
| `NormalizeObservation` | `test_normalize_observation.py` | none | **add** `test_normalize_observation_real_env` to `test_normalize_observation.py` |
| `PadObservation` | `test_pad_observation.py` | `test_obs_padding_invariants.py::test_pad_observation_keeps_non_zone_tail_at_stable_indices` already uses a real env via `fixture_registry` | **no duplicate** — add a one-liner to `test_obs_padding_invariants.py` docstring confirming it is the PadObservation companion test; no new test body needed |
| `wrap_env_for_rl` composition | `test_wrap_env_for_rl.py` | none | **add** `test_wrap_env_for_rl_real_env_invariant` to `test_wrap_env_for_rl.py` |

Each companion test: build env from `fixture_registry`, wrap, `reset()`, assert
one invariant (observation space dtype, shape, or metadata field shape matches
production env). ~10 lines per test.

---

## Item 7 – Stale fixture audit

Reference counts (via grep across `tests/`):

| fixture path | references outside `test_get_actuators.py` | decision |
|---|---|---|
| `tests/fixtures/bldg1.epjson` | 0 | **`git rm`** |
| `tests/fixtures/bldg1-setpoint-control/` | 0 | **`git rm -r`** |
| `tests/fixtures/bldg2.epjson` | 0 | **`git rm`** |
| `tests/fixtures/eplusout.edd` | only referenced by `test_get_actuators.py`, which is deleted (Item 1) | **`git rm`** |
| `tests/fixtures/eplustbl.htm` | 0 (no references anywhere in `tests/`) | **`git rm`** |
| `tests/fixtures/in.schedules.csv` | 0 | **`git rm`** |
| `tests/fixtures/weather.epw` (top-level) | `conftest.py:104` — that line is `weather_file="weather.epw"` inside a `BuildingInfo` constructor where `building_dir` is a `minimal_*` fixture subdir; it is a field *value*, not a file path. `test_pipeline_discovery.py:45` references `fixture_dir / "weather.epw"` inside a per-fixture subdir. Neither reads from `tests/fixtures/weather.epw` directly. The only direct filesystem read is in the dropped long test. | **`git rm`** |

### D10 deferred thermostat setpoints stub

`building2building/pipeline/steps/thermostat_setpoints.py` defines
`AddSetpointControl`, `add_setpoint_control`, `get_temperature_setpoints`.

Call-site audit:
```
grep -rn "AddSetpointControl\|add_setpoint_control\|get_temperature_setpoints" \
    building2building/ tests/ --include="*.py" \
  | grep -v "thermostat_setpoints.py\|pipeline/__init__.py"
```
Returns empty — no live call sites outside the module itself and its
`__init__.py` re-export.

**Decision:**
- `git rm building2building/pipeline/steps/thermostat_setpoints.py`
- Remove the three names from the `from … import (…)` block at line 50–53 of
  `building2building/pipeline/__init__.py`.
- Remove the three entries from `__all__` at lines 156–158.

---

## Item 8 – `test_default_path_constant_points_inside_package`

In `tests/quick/test_reward_normalizers.py::TestDefaultPathSentinel`.

`building2building/data/reward_normalizers.yaml` was last modified in commit
`9ab1044` (T-pre sweep), indicating the `data/` layout is still being touched.
The test guards against accidental removal of the committed YAML.

**Decision: keep.**

---

## Item 9 – Quick-tier timing gate

Cannot be pre-verified from the audit (requires pytest execution).

**T27b action:** run `pytest -m quick --durations=20` and record wall-clock.
For any single test > 10 s: split, move to `long/`, or add a justification
comment in the test docstring. Target: total ≤ 90 s.

Files originally flagged with "Audit note: scheduled for T27 move to `quick/`":

| file | action | rationale |
|---|---|---|
| `tests/long/test_observation_dimension.py` | **`git rm`** | `test_observation_names_stability.py` parametrizes over all three modes (`constant`, `occupancy`, `random_schedule`) and asserts exact observation names via snapshots, which subsumes every contract here.  Moving would require a non-trivial refactor (files uses `make_env_from_config` with real dataset buildings, not the minimal fixture env). |
| `tests/long/test_rescale_action.py` | **`git rm`** | `test_wrap_env_for_rl.py` covers the bounds and round-trip with `MockEnv`; the new `test_wrap_env_for_rl_real_env_invariant` companion verifies the same invariants on a real env.  Same refactor problem as above. |

---

## Item 10 – Module docstrings

Files with no module-level docstring (confirmed via `ast.get_docstring`):

```
tests/quick/test_augment_building_params.py
tests/quick/test_equipment_schema.py
tests/quick/test_make_env_cleanup.py
tests/quick/test_make_env_minimal.py
tests/quick/test_normalize_observation.py
tests/quick/test_observation_names_stability.py
tests/quick/test_obs_padding_invariants.py
tests/quick/test_pad_observation.py
tests/quick/test_patch_epjson_run_period.py
tests/quick/test_pipeline_discovery.py
tests/quick/test_pipeline_make_controllable.py
tests/quick/test_pipeline_prepare_building.py
tests/quick/test_resample_building_wrapper.py
tests/quick/test_wrap_env_for_rl.py
```

Additionally, after being moved to `quick/` (Item 9):
```
tests/quick/test_observation_dimension.py   (replace "Audit note" prefix)
tests/quick/test_rescale_action.py          (replace "Audit note" prefix)
```

**T27b action:** add a module docstring to each file above. The docstring must
state the *contract* the file pins (not "tests for module X"). The "Audit note"
prefix in the two moved files must be replaced with the actual invariant
already present in the rest of the docstring.

---

## Item 11 – `pytest.mark.skip` / `pytest.mark.xfail` without tracked TODO

```
grep -rn "pytest.mark.skip\b\|pytest.mark.xfail" tests/ --include="*.py"
```

Returns empty.

The two `pytest.mark.skipif` occurrences in `tests/long/test_env_leak.py`
(lines 132 and 251) are runtime-condition skips gated on `_PSUTIL_AVAILABLE`
(a third-party library). These are not unconditional `skip` or `xfail`
markers; they are appropriate defensive guards.

**Decision: no action.**

---

## Summary table for T27b

| # | files touched | action |
|---|---|---|
| 1a | `tests/test_get_actuators.py` | `git rm` — tests a dead export |
| 1b | `tests/test_wrappers.py` | `git rm` |
| 2 | `tests/long/test_pipeline_single_zone_houses.py` | `git rm` |
| 3 | `tests/quick/test_eval_bugs.py` → `tests/quick/test_eval_path_layout.py` | delete 3 classes, rename, update docstring |
| 4 | — | no action |
| 5 | — | no action |
| 6a | `tests/quick/test_augment_building_params.py` | add `test_augment_building_params_real_env` |
| 6b | `tests/quick/test_normalize_observation.py` | add `test_normalize_observation_real_env` |
| 6c | `tests/quick/test_obs_padding_invariants.py` | update module docstring only |
| 6d | `tests/quick/test_wrap_env_for_rl.py` | add `test_wrap_env_for_rl_real_env_invariant` |
| 7a | `tests/fixtures/bldg1.epjson`, `bldg2.epjson`, `eplusout.edd`, `eplustbl.htm`, `in.schedules.csv`, `weather.epw` | `git rm` |
| 7b | `tests/fixtures/bldg1-setpoint-control/` | `git rm -r` |
| 7c | `building2building/pipeline/steps/thermostat_setpoints.py` | `git rm` |
| 7d | `building2building/pipeline/__init__.py` | remove 3 imports + 3 `__all__` entries |
| 8 | — | no action |
| 9a | `tests/long/test_observation_dimension.py` | `git rm` — contracts subsumed by `test_observation_names_stability.py` |
| 9b | `tests/long/test_rescale_action.py` | `git rm` — contracts subsumed by wrapper tests + companion |
| 9c | — | run `pytest -m quick --durations=20`, fix any > 10 s outliers |
| 10 | 14 + 2 files listed in Item 10 | add / rewrite module docstrings |
| 11 | — | no action |
