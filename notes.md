<!-- -*- mode: markdown -*- -->

# Notes

Compact, in-flight working notes. Read this **before** starting any task.
Update as the project moves; keep it short and context-efficient.

Companion: `design_doc.md` (slow-changing intent + roadmap),
`TODO.md` (atomic action items). The LaTeX paper (final results) is
maintained outside this repo.

---

## Reward — calibration regime and impl map

Calibration regime (load-bearing for the reward-coefficient study and
the paper rerun):

- Calibration task: `task_occ_*` family (occupancy-driven setpoints,
  seasonal unoccupied: winter 18 °C / shoulder 21 °C / summer 26 °C),
  `dT = 1.0`.
- Run period: full year by default; YAML supports per-season slicing
  (`winter`, `summer`, `full_year`).
- Split: train.
- Aggregation: median across buildings within each
  `(building_type, climate_zone)` bucket. `SingleFamilyHouse` is one
  bucket (key `cz0`, no ASHRAE CZ assignment).

Task family: nine presets `task_<mode>_<level>` over the 3×3 grid.
mode ∈ {`const`, `occ`, `rand`}; level ∈ {`e0`, `emed`, `ehigh`}
with `w_E ∈ {0.0, 1.0, 5.0}` (final `emed`/`ehigh` pinned by Phase B).

Implementation map:

| Concern | Location |
| --- | --- |
| Reward config dataclass (sentinel pattern) | `building2building/types.py::NormalizedDeadbandRewardConfig` |
| Reward function | `building2building/simulator/rewards.py::NormalizedDeadbandReward` |
| Calibration constants loader | `building2building/data/reward_normalizers.py` |
| Calibration YAML | `building2building/data/reward_normalizers.yaml` |
| Task presets + factory | `building2building/config/tasks.py` |
| Env-factory auto-fill of `(τ_T, τ_E)` | `building2building/api/__init__.py::new_make_env` |
| Calibration scripts | `analysis/task_study/compute_reward_normalizers.py`, `compute_random_policy_reward_normalizers.py` |

Sanity figure (git-ignored):
`analysis/task_study/reward_design/plots/fig_random_policy_normalizer_calibration.png`.
Regenerate with
`python -m analysis.task_study.compute_random_policy_reward_normalizers --mode aggregate`
(needs the rollout cache under `$SCRATCH/b2b_reward_normalizers_random/data/`).
Per-bucket median of `temp_penalty/τ_T` and `power_penalty/τ_E` is 1.0
by construction; bucket IQRs match the `tau_*_iqr` fields in the YAML.

---

## Reward family — implementation gotchas

- `NormalizedDeadbandRewardConfig` has **two valid states**: unfilled
  sentinel (`tau_T = tau_E = None`, used by presets) and filled
  (both floats > 0, ready for the simulator). Mixed state is always
  a bug — the dataclass `__post_init__` raises.
- `lru_cache` on `_cached_load` (capacity 12, keyed by path +
  run_period). Tests that mutate the YAML must call
  `clear_reward_normalizers_cache()`.
- Reward decomposition for analysis: do not reuse the legacy
  `temp = -reward - w * power` formula on normalized runs. Use
  both `tau_T` and `tau_E`. Checkpoint `.npz` files written by
  `_eval_and_save` carry the tau values for this reason.
- `task_occ_*` is the on-calibration regime by construction.
  `task_const_*` and `task_rand_*` reuse the same `(τ_T, τ_E)` (these
  are calibration constants per `(bt, cz)`, not per task), so off-mode
  usage is approximate; the simulator emits a `RuntimeWarning` (deduped
  per `(bt, bid, mode, dT)`).
- Tail buildings drift from `w_E=1` semantics: at the 25th/75th
  percentile of a loose bucket the effective `w_E` can be off by
  30–60%. Cross-bucket comparisons should report bucket and
  percentile when possible.

---

## RL env wrapper stack

All RL training and eval (PPO, SAC, dynamics adaptation, cross-domain)
go through `make_rl_env_fn` (`baselines/utils/training.py`). The wrapper
stack is exactly:

```
Monitor ( NormalizeObservation ( RescaleAction ( TimeLimit ( EnergyPlusSimulator ) ) ) )
```

- `NormalizeObservation` is **deterministic** (`[0, 1]` bounds from
  `simulator/observation_spaces.py::flat_observation_info`). No
  `VecNormalize` stats file. The wrapper does **not** clip; values
  outside `[0, 1]` during warmup are expected.
- `RescaleAction(-1, 1)` is applied **inside** `new_make_env` (with
  `rescale_action=True`); `wrap_env_for_rl` is then called with
  `rescale_action=False` to avoid double-rescaling.
- Reactive controllers, tuning scripts, and `train_cross_domain.py`
  (Amorpheus) bypass this stack and use raw engineering units.
- `train_dynamics_adaptation.py::train_specialist` is the one
  exception that does not currently apply `NormalizeObservation`. If
  Phase A reruns the specialist path, add an explicit
  `wrap_env_for_rl` call there.

---

## Training stability notes

No further PPO/SAC hyperparameter sweeps before the camera-ready rerun
(decision 2026-05-25). Final agents use the committed defaults.

**SAC** (`baselines/configs/policy/sac.yaml`): under the normalized
reward SAC tends to chatter, drift toward action boundaries, and
degrade on deterministic eval after an early peak. Stabilized by the
committed B1 fixes — `log_std_init=0.0` (was −3.0, which kept initial
post-tanh actions in [−0.05, 0.05]), `ent_coef=0.2` (was `auto`, which
mis-tunes against the small post-tanh entropy once actions saturate),
`use_sde=false` (gSDE noise reinforced boundary pushing). Occupancy
setpoints + 5-min cadence are intrinsically hard for SAC; if the cheap
fixes prove insufficient, consider forward-looking schedule features,
an `||a_t − a_{t-1}||²` action-rate penalty, or a lower control cadence.

**PPO** (`baselines/configs/policy/ppo.yaml` as committed): freezes
after the first gradient update at high `w_E` on `winter` (1/6
buildings at `w_E=1`, 6/6 at `w_E=100`) — the initial gradient is
dominated by the power axis, saturates an action channel, and
`target_kl=0.02` prevents recovery. Single-seed, winter-only, 6
buildings; not reproduced on `full_year` (the calibration regime).
Safe range on winter: `w_E ∈ {0, 1, 5}`.

**Cross-checks for the reruns:** `task_const_*` is a useful confounder
control — if constant-target SAC is materially more stable than
occupancy SAC at the same `w_E`, occupancy is the destabiliser, not the
reward. Old `.npz` checkpoints under
`$SCRATCH/b2b_reward_design_normalized/data/` are not comparable to new
ones (per-step reward distributions differ); delete before any new
comparison sweep.

---

## Operational gotchas (cluster / SLURM)

- `sbatch --wrap='... source .venv/bin/activate ...'` runs under
  `/bin/sh` (no `source`). Use a real `.sh` launcher with `#!/bin/bash`.
- `_eval_and_save` skips rewriting existing `.npz` files; changing the
  eval cadence retroactively produces a Frankenstein grid — delete the
  old files first.
- `SubprocVecEnv` workers are single-threaded EnergyPlus; `N_ENVS =
  cpus − 2` is the sweet spot (14 envs on 16 CPUs).
- `task_occ_e0` has `w_E = 0`, so the dominance-ratio plot is NaN by
  construction — expected, not a bug.

**EnergyPlus resource leak (B0) — fixed.** The leak-free lifecycle
lives on `minergym.environment.EnergyPlusEnvironment` itself
(`pyproject.toml` pinned to upstream `956c3e1`); all callers use plain
`env.close()`. Regression coverage: `tests/long/test_env_leak.py`.
**Residual ~14 MB/cycle is irreducible** — EnergyPlus C++ globals that
live outside `EnergyPlusData` and cannot be freed from Python; the only
complete fix is subprocess isolation. A 500-episode job accumulates
~7 GB; budget node memory accordingly.

Watch-list (not covered by automated tests):

- "EnergyPlus thread did not exit within Xs" in `.err` logs → the
  thread leaked in a C-level call (large buildings, CPU contention, or
  a simulation killed mid-step). Mitigation: raise `thread_join_timeout`.
- `$SLURM_TMPDIR` growing during a run → a crash between
  `make_energyplus()` and the first `reset()`, or `close()` never
  called (SIGKILL), leaves the output dir behind.
- RSS growing faster than ~20 MB/episode → the Python-side fix likely
  regressed (missing `self.ep = None` / `gc.collect()`, or a wrapper
  holding a reference to the old env). Run `test_env_leak.py` on an EP
  node to isolate.

---

## OfficeMedium OA-mixer fix (Phase M — done)

OfficeMedium agent action dim is **36** (was 33).
`make_vav_system_controllable` (`building2building/pipeline/actuators.py`)
now emits **3 outdoor-air mass-flow actuators** (one per air loop),
enumerated dynamically from the epJSON ontology
(`AirLoopHVAC → AirLoopHVAC:OutdoorAirSystem → Controller:OutdoorAir`),
not hardcoded. Key decisions (full rationale in the Phase M git
history):

- Actuator: `(Outdoor Air Controller, Air Mass Flow Rate)`, units
  kg/s, range **[0.0, 5.0]** (headroom above design flow without
  inflating the exploration range). `Availability Status` is **not**
  exposed (deliberate divergence from RL2GNNs); every actuator's
  enable-side is pinned on so agent writes are authoritative — asserted
  in `tests/quick/test_officemedium_actuator_set.py`.
- Reactive controller pins the OA mixer at a constant **1.37 kg/s** per
  loop (DOE autosized minimum ventilation), not Optuna-searched —
  learning the economizer is the agent's job. Field `oa_mass_flow` on
  `baselines/controllers/air_loop.py::AirLoopConfig`.
- **No backward compatibility.** The regenerated OfficeMedium slice is
  on `vtaboga/building2building_dataset@main`; a stale `equipment.json`
  (missing `oa_mass_flow`) fails `cattrs.structure` loudly. Running an
  arbitrary historical commit against `main` HF is unsupported.
- M4 rewrote **every row** of `reward_normalizers.yaml`, not just
  OfficeMedium: the calibration scripts had hard-coded the deleted
  `task3` preset and recovered `temp_penalty` by inverting the
  un-normalized reward formula (invalid against
  `NormalizedDeadbandReward`). They now pass `task_occ_emed` and
  recompute `(temp_penalty, power_penalty)` directly from
  `info["raw_observation"]` via
  `simulator.rewards._deadband_components`.

---

## D10 — pyright status

`pyright building2building baselines` reports ~242 pre-existing errors,
**none in the public API surface** (`building2building/__init__.py.__all__`
is clean). Left as-is for the OSS release. Root causes by frequency:
matplotlib/torch private-import stubs (~80); Hydra `str` →
`BuildingType` Literal coercion in baselines (~40); `PPO` not
assignable to the `PolicyLike` protocol (~5); scattered `cattrs` /
untyped-object gaps in `pipeline/`, `sources/`, `store.py`,
`simulator/wrappers.py` (~120).
