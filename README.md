# Building2Building

[![arXiv](https://img.shields.io/badge/arXiv-2607.16534-b31b1b.svg)](https://arxiv.org/abs/2607.16534)
[![Docs](https://readthedocs.org/projects/building2building/badge/?version=latest)](https://building2building.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of *"Building2Building: A Large Scale Benchmark for
Generalizable Real-World Reinforcement Learning"* (Taboga, Veilleux, Jang,
Rankawat, and Bacon; Reinforcement Learning Journal, 2026).

> **Reproducing the paper.** All results reported in the paper were produced
> with [`v1.0.0`](https://github.com/vtaboga/building2building/releases/tag/v1.0.0).
> `main` continues to evolve — check out the tag for exact reproduction:
>
> ```bash
> git clone https://github.com/vtaboga/building2building.git
> cd building2building && git checkout v1.0.0
> ```

Building2Building (B2B) is a large-scale reinforcement learning benchmark for
HVAC control in buildings. It exposes 6,000 parametrically generated
EnergyPlus building models as Gymnasium environments, spanning 6 building types,
16 ASHRAE climate zones, and 3 HVAC system types.

B2B is designed to accelerate research in **transfer learning**, **multi-task
RL**, and **meta-learning** for building energy management.

**Key features:**

- 6,000 buildings (5,400 train / 600 test) from ASHRAE 90.1-2022 prototypes and residential archetypes
- 6 building types: `SingleFamilyHouse`, `OfficeSmall`, `OfficeMedium`, `RetailStandalone`, `RestaurantFastFood`, `Warehouse`
- 6 named task presets (3×2 grid over setpoint mode × energy weight)
- 4 benchmark problems testing dynamics adaptation, cross-domain generalization, goal adaptation, and action-space transfer
- Normalized scoring against reactive-controller baselines
- Morphology graph for structured observation/action decomposition
- Gymnasium-compatible with full SB3 integration
- Pre-processed buildings downloadable from HuggingFace

---

## Installation

```bash
pip install -e .                          # Core (environments only)
pip install -e ".[training]"              # + PyTorch, SB3, Hydra, Optuna, W&B
pip install -e ".[all]"                   # Everything (training + test + dev + docs)
```

EnergyPlus 24.1 is required and will be automatically downloaded. See the documentation for more details.

---

## Quick Start

```python
import building2building as b2b

# List available building types
print(b2b.list_building_types())
# ['SingleFamilyHouse', 'OfficeSmall', 'OfficeMedium', ...]

# Create a Gymnasium environment
env = b2b.make_env("OfficeSmall", split="train", index=0, task="task_const_e0")

obs, info = env.reset()
done = False
total_reward = 0.0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode return: {total_reward:.2f}")
env.close()
```

---

## Core API

Everything below is part of the stable, public `building2building` surface
(see [`docs/api/stability.md`](docs/api/stability.md)).

```python
import building2building as b2b

# --- Discovery -------------------------------------------------------------
b2b.list_building_types()                     # 6 building-type names
b2b.list_buildings("OfficeSmall", split="train")        # 900 train IDs
b2b.list_buildings("OfficeSmall", split="test")         # 100 held-out IDs
b2b.list_buildings("OfficeSmall", split="test_small")   # 8-building quick split
b2b.get_climate_zone("OfficeSmall-0001")      # ASHRAE climate zone

# --- Environment creation --------------------------------------------------
# By split + index (downloads the building from HuggingFace on first use):
env = b2b.make_env("OfficeSmall", split="test", index=0, task="task_const_e0")
# Or by explicit building ID:
env = b2b.make_env("OfficeSmall", building_id="OfficeSmall-0001",
                   task="task_occ_e05", run_period="full_year")

# Per-env metadata (names, morphology graph) lives on env.metadata:
obs_names = env.metadata["observation_names"]
act_names = env.metadata["action_names"]
morphology = env.metadata["morphology"]       # structured obs/action decomposition

# --- Rollouts --------------------------------------------------------------
@b2b.callable_controller
def random_controller(obs):
    return env.action_space.sample()

traj = b2b.rollout(env, random_controller)     # -> Trajectory(obs, actions, rewards, ...)

# --- RL training helper ----------------------------------------------------
rl_env = b2b.wrap_env_for_rl(env)              # SB3-ready (flattened, normalized)

# --- Normalized scoring ----------------------------------------------------
score = b2b.compute_normalized_score(
    cumulative_return=traj.rewards.sum(),
    building_type="OfficeSmall", task="task_const_e0",
    run_period="full_year", building_id="OfficeSmall-0001",
)                                              # lower is better; score < 1.0 beats the reactive baseline
```

Wrappers for multi-building training (zero-pad, normalize, augment with
building parameters, resample on reset) are exported at the top level:
`b2b.PadObservation`, `b2b.NormalizeObservation`,
`b2b.AugmentObservationWithBuildingParams`, `b2b.ResampleBuildingOnResetWrapper`.

---

## Building Types

| Type | Zones | HVAC System | Description |
|---|---|---|---|
| `SingleFamilyHouse` | 1 | Unitary | Residential single-zone houses |
| `OfficeSmall` | 5 | Unitary | Small office prototype |
| `OfficeMedium` | 15+ | VAV air-loop | Medium office prototype |
| `RetailStandalone` | 4 | Unitary | Standalone retail store |
| `RestaurantFastFood` | 2 | Unitary | Fast food restaurant |
| `Warehouse` | 3 | Unitary | Warehouse prototype |

Each type contributes **1,000 parametric instances**, for **6,000 buildings** total.

### Dataset Splits

Every building type ships with three splits, selected via the `split=` argument
to `make_env` / `list_buildings`:

| Split | Per type | Total | Use |
|---|---|---|---|
| `train` | 900 | 5,400 | Training |
| `test` | 100 | 600 | Held-out evaluation |
| `test_small` | 8 | 48 | Fast evaluation / smoke tests (subset of `test`) |

---

## Task Presets

Six named presets form a 3×2 grid over setpoint mode and energy weight.
Comfort is unnormalized (`tau_T = 1`, raw °C²); energy is normalized by the
per-bucket reference-controller spend (`tau_E`), so `energy_weight` is a
dimensionless price with the same meaning in every building:

| Task | Mode | Energy Weight |
|---|---|---|
| `task_const_e0` | Constant | 0.0 (comfort-only) |
| `task_const_e05` | Constant | 0.5 (energy priced in) |
| `task_occ_e0` | Occupancy (seasonal) | 0.0 |
| `task_occ_e05` | Occupancy (seasonal) | 0.5 |
| `task_rand_e0` | Random schedule | 0.0 |
| `task_rand_e05` | Random schedule | 0.5 |

The default is `task_const_e0` (constant setpoint, comfort-only).

```python
env = b2b.make_env("OfficeSmall", task="task_occ_e05")
```

---

## Benchmark Problems

Each benchmark class tests a different generalization axis:

| Class | What varies | What stays fixed |
|---|---|---|
| `DynamicsAdaptation` | Building dynamics (different instances) | Reward, action space |
| `CrossDomainGeneralization` | Building type (train type A, test type B) | Reward, action space |
| `GoalAdaptation` | Reward / task | Building, action space |
| `ActionSpaceTransfer` | Controllable actuators | Building, reward |

```python
bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task_const_e0")
train_ids = bench.train_building_ids()
test_ids = bench.test_building_ids()
train_envs = bench.make_train_envs(n=4)
test_envs = bench.make_test_envs(n=4)
```

---

## Baselines

Reference experiment scripts live in `baselines/` and use only the public
`building2building` API. They are [Hydra](https://hydra.cc/)-configured — every
value (`seed`, `tasks`, `building_types`, `training.total_timesteps`, …) is
overridable on the command line. Install with `pip install -e ".[training]"`.

**Main scripts** (one canonical command each):

```bash
# Reactive controllers — evaluate and (re)generate baseline_returns.csv
python -m baselines.run_reactive_control experiment=eval_reactive_control

# Per-building specialists (paper §5)
python -m baselines.train_ppo experiment=train_ppo
python -m baselines.train_sac experiment=train_sac

# Dynamics adaptation (paper §6.1): specialist | baseline | parameterized
python -m baselines.train_dynamics_adaptation experiment=train_dynamics_parameterized difficulty=easy

# Cross-domain transfer with the Amorpheus transformer (paper §6.2)
python -m baselines.train_cross_domain experiment=train_cross_domain

# Hyperparameter tuning (Optuna)
python -m baselines.tune_controller experiment=tune_controller building_type=OfficeSmall
python -m baselines.tune_ppo experiment=tune_ppo task=task_const_e0
```

**Evaluation & plotting:**

```bash
python -m baselines.eval_ppo --model-dir outputs/train_ppo/models
python -m baselines.eval_dynamics_adaptation --model-path <model.zip> --difficulty easy --approach parameterized
python -m baselines.eval_cross_domain --model-path <model.pt> --test-building-types Warehouse SingleFamilyHouse

python -m baselines.plotting.plot_ppo_specialist --ppo-csv results_ppo.csv --baseline-csv baseline_returns.csv
python -m baselines.plotting.plot_dynamics_adaptation ...
python -m baselines.plotting.plot_cross_domain --results-csv results_cross_domain.csv
```

Common overrides — subset to a quick run, or sweep the full task grid:

```bash
# Quick smoke run
python -m baselines.train_ppo experiment=train_ppo \
    building_types=[OfficeSmall] tasks=[task_const_e0] training.total_timesteps=100_000

# Full 9-task grid on the fast split
python -m baselines.train_ppo experiment=train_ppo_task_study \
    --multirun seed=0 building_split=test_small
```

See [`baselines/README.md`](baselines/README.md) for the full reference and
[`docs/baselines/`](docs/baselines/overview.md) for per-experiment guides.

---

## Normalized Scoring

Score an agent relative to the reactive-controller baseline:

```python
score = b2b.compute_normalized_score(
    cumulative_return=-5000.0,
    building_type="OfficeSmall",
    task="task_const_e0",
    run_period="full_year",
    building_id="OfficeSmall-0001",
)
# lower is better: score < 1.0 means the agent outperforms the baseline
```

> **Note:** the packaged baseline file can be regenerated by running
> `python -m baselines.run_reactive_control experiment=eval_reactive_control`.

---

## Documentation

Full documentation: **https://building2building.readthedocs.io/**

---

## Tests

```bash
pip install -e ".[test,training]"         # test + training deps (the quick suite imports baselines)
pytest -m quick                           # Fast tests (no EnergyPlus)
B2B_RUN_LONG_TESTS=1 pytest -m long      # Simulation-heavy tests
B2B_RUN_LONG_TESTS=1 pytest              # Full suite
```

---

## Citation

If you use Building2Building in your research, please cite:

```bibtex
@article{taboga2026building2building,
  title   = {Building2Building: A Large Scale Benchmark for Generalizable
             Real-World Reinforcement Learning},
  author  = {Taboga, Vincent and Veilleux, Justin and Jang, Doseok and
             Rankawat, Anushree and Bacon, Pierre-Luc},
  journal = {Reinforcement Learning Journal},
  year    = {2026},
  eprint  = {2607.16534},
  archivePrefix = {arXiv},
}
```

GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff),
which also records the software version.

---

## License

MIT — see [`LICENSE`](LICENSE).


