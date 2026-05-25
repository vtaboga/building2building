<!-- -*- mode: markdown -*- -->

# Building2Building — Design Doc

Stable project reference. Slow-changing.

Companions: `AGENTS.md` (working principles), `notes.md` (in-flight
state, decisions, gotchas), `TODO.md` (atomic action items),
`paper/main.tex` (authoritative for problem statement and final
results), `paper/reviewer_feedback.md`.

---

## 1. Intent

B2B is a large-scale RL benchmark for HVAC control; the object of
study is **generalization in RL** in a heterogeneous, high-dimensional,
slow-simulating real-world domain. Three intertwined questions:

1. **Multi-task / meta-RL** — one policy across thousands of related
   buildings (cross-domain benchmark; Amorpheus).
2. **Transfer** — across dynamics, goal, or action-space changes
   (three transfer benchmarks).
3. **Scale** — ~7,000 envs; which conclusions survive the jump.

A change that helps one at the cost of another is rarely worth it.

**Audience.** Primary: ML researcher who never opens EnergyPlus.
B2B is a `pip install` Gymnasium catalogue; every EnergyPlus concept
is hidden or documented as a hyperparameter. Secondary: building-energy
researcher extending the benchmark via `building2building/pipeline/`
(intentionally separate from the user-facing surface).

### Benchmark problems (authoritative defs in `paper/main.tex`)

| Problem | Varied | Held constant |
| --- | --- | --- |
| Cross-domain generalization | archetype, climate zone, HVAC family | goal, action topology |
| Dynamics adaptation | physics within an archetype family | goal, action topology |
| Goal adaptation | setpoint schedule (occ/rand/const) | building, action |
| Action-space transfer | action topology | building, goal |

---

## 2. Design principles

- **API is the abstraction barrier.** `building2building/api/` speaks
  Gymnasium/NumPy/Python. South of it: EnergyPlus, .idf, eppy,
  schedules. The API may not leak `.idf` paths, EP version strings,
  node-type enums, actuator handles. Test: would a Meta-World user
  expect to see this?
- **Hackability over extensibility.** Straight-line code over
  abstractions; concrete types over generic; one module, one concern;
  comments explain *why*, never *what*. Plugins/registries only when a
  concrete second use case exists.
- **Reproducibility over efficiency.** Single-process EP over threaded;
  deterministic obs normalization (`observation_space.low/high`) over
  running stats; committed `baseline_returns.csv` over regenerate;
  concrete seeds over `null`. Every shipped number reproducible from
  one command — see [`REPRODUCING.md`](REPRODUCING.md) (Phase D).
- **Fail loudly** (per `AGENTS.md`). No silent fallbacks, no swallowing
  try/except. Sole sanctioned non-fatal signal: the off-calibration
  `RuntimeWarning` in `simulator/__init__.py` (deduped, never
  swallowed).

---

## 3. Architecture

### 3.1 Layout

```
building2building/        # Public package
├── api/                  # new_make_env, list_buildings, rollout, RL wrappers
├── benchmarks/           # 4 benchmark problem classes
├── config/               # TaskPreset + dataclass parsers
├── data/                 # HF dataset registry + reward_normalizers
├── envs/                 # Gymnasium registration + factory
├── morphology.py         # Structured graph rep (per-node spaces)
├── pipeline/             # EnergyPlus building processing (offline)
├── scoring.py            # compute_normalized_score
├── simulator/            # Reward, obs, action, schedule modules
├── sources/              # Building-archetype loaders
├── scores/               # baseline_returns.csv
└── types.py              # Domain dataclasses + reward configs

baselines/                # Paper experiments (use only public API)
├── controllers/, configs/ (Hydra), train_*.py, eval_*.py, tune_*.py,
├── plotting/, utils/     # make_rl_env_fn lives in utils/training.py
tests/, tutorials/, docs/, paper/
```

Working/scratch (gitignored, retained as tripwires per
`notes.md` § Repo hygiene): `analysis/` (load-bearing — see notes),
`scripts/`, `summaries/` (slated for deletion). Do not recreate:
`scrap/`, `staging/`, `plans/`, `refactoring/`, `wandb/`, `site/`,
`logs/`, `.venv/`.

### 3.2 API stability tiers

- **Public (stable).** Re-exports from `building2building/__init__.py`
  (`new_make_env`, `list_buildings`, `list_building_types`,
  `list_buildings_by_climate_zone`, `get_climate_zone`,
  `wrap_env_for_rl`, `compute_normalized_score`, `types.py`
  dataclasses, morphology helpers). Contract — see §4. Breaking
  changes: deprecation cycle + `CHANGELOG.md` entry.
- **Public-but-research (semi-stable).** `baselines/` package + Hydra
  configs. CLI behaviour is contract; internal helpers are not.
- **Internal (no guarantee).** Everything else (`pipeline/`,
  `sources/`, `simulator/` internals, `analysis/`).

### 3.3 Key invariants

- `NormalizedDeadbandRewardConfig` has two valid states: unfilled
  sentinel (`tau_T=tau_E=None`, used by presets) and filled (both
  `>0`). Mixed → `__post_init__` raises. `new_make_env` resolves the
  per-bucket constants when it knows the building.
- `make_rl_env_fn` (`baselines/utils/training.py`) is the **single**
  way to build an env for RL training/eval. Wrapper stack:
  `Monitor(NormalizeObservation(RescaleAction(TimeLimit(simulator))))`.
- `NormalizeObservation` is **deterministic** (uses
  `observation_space.low/high`). No `VecNormalize` stats file.
- Reactive controllers, tuning, benchmark harnesses use
  `new_make_env(rescale_action=False)` (raw engineering units).
- Off-calibration `(building_type, target_mode, dT)` usage emits a
  deduped `RuntimeWarning` from
  `simulator/__init__.py::_maybe_warn_normalized_deadband`.

---

## 4. Testing

Three tiers, intentionally separated:

| Tier | What | Where | Runs EP? | CI |
| --- | --- | --- | --- | --- |
| 1 — API contract | Pins the §3.2 public surface | `tests/quick/test_api*.py`, `test_new_api.py`, `test_gym_registration.py`, `test_climate_zones.py`, `test_data_registry.py`, `test_selection_and_env_creation.py`, `test_types.py` | No (fixtures) | Every PR |
| 2 — Pure logic | Reward, normalizers, scoring, schedules, wrappers | `tests/quick/test_*.py` (rest) | No | Every PR |
| 3 — Integration | End-to-end EP on a small subset | `tests/long/*` | Yes | Nightly / pre-release |

Tier-1 changes = deliberate API change ⇒ require `CHANGELOG.md` entry.
Hydra config changes use 50-step smoke configs in
`tests/quick/test_train_*_smoke.py`. Targets: Tier 1+2 < 5 min;
Tier 3 < 1 h. (Phase D7 wires GitHub Actions.)

---

## 5. Reward (secondary contribution)

The normalized deadband reward is both (a) a **precondition for the
generalization claims** — without per-bucket calibration, `w_E` does
not mean the same thing across buildings (per-zone temperature error
and HVAC Wh/m² differ by ~10–40× across the catalogue), so cross-
building generalization is uninterpretable — and (b) a **methodological
contribution in its own right**: a recipe for making a multi-objective
reward comparable across an environment family. The recipe (any
bucketing + any reference controller + any two-term reward) is
general; we instantiate it with the building taxonomy and a deadband
reward.

This is **not** the core contribution (the benchmark + three
generalization questions are), but deserves its own paper box and
docs page (TODO D14).

### Formula

For each step:

\[
r = -\Big(\frac{\text{temp\_penalty}}{\tau_T}
        + w_E \cdot \frac{\text{power\_penalty}}{\tau_E}\Big)
\]

`temp_penalty` is per-zone-averaged squared deviation from the target
(deadband shape preserved). `power_penalty` is HVAC electricity + gas
in Wh/m². The two `(τ_T, τ_E)` are calibrated per
`(building_type, climate_zone)` so that, at the median bucket
building under a reference controller, each contribution sits near
1.0. After normalization, `w_E` is dimensionless: `<1` favours
comfort, `≈1` balanced, `>1` energy.

Calibration regime (controller choice, run periods, YAML files,
implementation map): `notes.md` § "Reward — calibration regime and impl map".

---

## 6. Roadmap

Four phases. Hard cross-phase constraint: **D2 (legacy reward
deletion) precedes C1 (regen `baseline_returns.csv`)**. Inside a
phase, items are reorderable. Atomic items: `TODO.md`. In-flight
status: `notes.md`.

- **Phase A — Finalize the reward.** ✓ Calibration controller chosen:
  `reward_normalizers.yaml` (SAC-warmup uniform-random policy).
  `(τ_T, τ_E)` are locked; downstream re-runs can proceed.
- **Phase B — Stabilize RL training.** B0 first: fix EnergyPlus
  resource leak. Then reconcile SAC config, re-tune PPO `target_kl`,
  validate on `full_year` (not winter).
- **Phase C — Re-run paper experiments.** Regen
  `baseline_returns.csv`, then PPO specialists, dynamics adaptation,
  cross-domain Amorpheus. Update `paper/main.tex` figures/tables
  (camera-ready scope rule per `AGENTS.md`).
- **Phase D — OSS readiness.** Two parallel tracks:
  - **D-API**: freeze public surface, label Tier-1 contract tests,
    `REPRODUCING.md`, tutorial/docs migration off legacy tasks,
    benchmark-problem docs pages.
  - **D-house**: LICENSE, CI, legacy reward deletion (D2 — precedes
    C1), README refresh, `black` + `pyright`, drop
    `requirements.txt`.

---

## 7. Out of scope

- New archetypes, climate zones, HVAC families, benchmarks, or
  algorithms beyond PPO + SAC + reactive + Amorpheus.
- Backward compatibility with the un-normalized reward family.
- Refactoring `simulator/`, `morphology.py`, `pipeline/`, or the
  benchmarks beyond what the reward redesign requires.
- Re-tuning reactive controllers (would invalidate
  `baseline_returns.csv` mid-rerun).
- Production-grade software engineering. Optimizing for HVAC
  practitioners as primary audience.
