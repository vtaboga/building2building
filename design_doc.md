<!-- -*- mode: markdown -*- -->

# Building2Building — Design Doc

Stable project reference. Slow-changing.

Companions: `AGENTS.md` (working principles), `notes.md` (in-flight
state, decisions, gotchas), `TODO.md` (atomic action items). The
LaTeX paper (authoritative for problem statement and final results)
is maintained outside this repository.

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

### Benchmark problems (authoritative defs in the paper)

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
  one command — see [`REPRODUCING.md`](REPRODUCING.md).
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
tests/, tutorials/, docs/
```

Working/scratch (gitignored): `analysis/` (load-bearing — dev
scratchpad whose paper-cited scripts migrate into `baselines/`),
`scripts/`. Do not recreate:
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

Three tiers, separated by **rollout length** — not EnergyPlus
availability (EP is a hard dependency, wired up once at collection).
Per-file inventory and design principles: `docs/about/testing.md`.

| Marker | What | Gate |
| --- | --- | --- |
| `quick` | No / short rollout (≤ ~20 steps); EP + cached HF fixtures allowed. Each test a few seconds. | Every PR (CI) |
| `long` | Multi-day rollouts, or per-cycle leak / lifecycle iteration. | `B2B_RUN_LONG_TESTS=1` |
| `release` | Dataset / artifact integrity vs the published HF dataset. | `tests/release/`, opt-in `-m release` |

Public-surface (§3.2) changes are deliberate API changes ⇒ require a
`CHANGELOG.md` entry. CI (GitHub Actions) runs `pytest -m quick` on
every push.

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
docs page.

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

The reward is finalized (`reward_normalizers.yaml`; `(τ_T, τ_E)`
locked) and the OSS-readiness phases are **complete**: M (OfficeMedium
OA-mixer fix), D (LICENSE, CI, legacy-reward deletion, `REPRODUCING.md`,
docs/tutorial migration, benchmark docs pages), T (test suite), F
(file/doc audit). Two phases remain — atomic items in `TODO.md`,
in-flight status in `notes.md`:

- **Phase A — Re-run paper experiments** on the final OSS codebase:
  regen `baseline_returns.csv`, PPO + SAC specialists, dynamics
  adaptation, cross-domain Amorpheus. Produces the paper figures/CSVs.
- **Phase B — Empirical reward-coefficient study:** pin the final
  `emed` / `ehigh` energy weights via a SAC policy ladder and validate
  invariance across `test_small`. **B precedes A** (A consumes the
  locked coefficients).

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
