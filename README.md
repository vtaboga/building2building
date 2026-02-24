### Building2Building (B2B)

Building2Building is a **reinforcement-learning framework for EnergyPlus building simulation** built on top of **Gymnasium**. It provides:

- **A simulator environment** wrapping EnergyPlus via `minergym`
- **A data/pipeline layer** to select and prepare buildings (e.g., Hydro-Québec datasets)
- **Baselines**: rule-based HVAC controllers and Stable-Baselines3 training utilities
- **Benchmarks**: reproducible evaluation loops and experiment definitions on fixed building sets

---

### Quick start

#### Prerequisites

- **Python**: 3.10+
- **EnergyPlus**: automatically managed by the project (see notes below), or you can point to an existing install.

#### Setup

Create and activate a virtual environment, then install the project:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

#### EnergyPlus management (important)

By default, B2B will download and cache an EnergyPlus binary the first time it needs one. You can also provide an existing install:

- **Use your own EnergyPlus**: set `ENERGYPLUS_PATH` to the install directory.
- **Control cache location**: set `STORE_PATH` to where downloads/extractions should live.

---

### Running scripts

All scripts use **Hydra** configs from `configs/`. Hydra creates a run directory and (by default) changes the working directory to it; files written via `Path.cwd()` typically land inside that run directory.

Always activate the venv first:

```bash
source .venv/bin/activate
```

---

### Benchmark: adaptive dynamics (Hydro-Québec building set)

This benchmark evaluates a **controller policy** (e.g. `unitary_sat`) across a fixed list of Hydro-Québec buildings. The default script config is `configs/bm_adaptive_dynamics.yaml`.

#### Minimal smoke run (fast)

```bash
python -m scripts.bm_adaptive_dynamics benchmark.limit=1 benchmark.max_steps=1
```

#### Typical usage

- **Run on the whole split**:

```bash
python -m scripts.bm_adaptive_dynamics benchmark.split=train benchmark.start=0 benchmark.limit=0
```

- **Evaluate a different controller** (examples):

```bash
python -m scripts.bm_adaptive_dynamics policy=unitary_pi
python -m scripts.bm_adaptive_dynamics policy=fan_coil_constant
```

Outputs are stored under:

- `outputs/benchmarks/adaptive_dynamics/<date>/<time>/`

The benchmark also writes incremental results to:

- `adaptive_dynamics_results.jsonl`

---

### Baseline rollouts: `scripts/baselines.py`

`scripts/baselines.py` runs **rule-based baseline controllers** (not SB3 training) and saves a rollout as CSV/NPZ.

#### Run the default baseline config

```bash
python -m scripts.baselines
```

This uses `configs/baseline.yaml`, which composes:

- building selector: `configs/bldg/*`
- controller policy: `configs/policy/*` (must define `policy.type`)
- reward: `configs/reward/*`

#### Useful overrides

- **Choose a controller policy**:

```bash
python -m scripts.baselines policy=unitary_sat
python -m scripts.baselines policy=unitary_pi
python -m scripts.baselines policy=fan_coil_constant
```

- **Change rollout length**:

```bash
python -m scripts.baselines env.max_steps=672 n_episodes=1
```

Rollout artifacts are written to the Hydra run directory (example files):

- `rollout.csv`
- `rollout.npz`
- `config_resolved.json`

---

### RL training entrypoint: `scripts/main.py` (Hydro-Québec)

`scripts/main.py` is the **Stable-Baselines3 training entrypoint** for the Hydro-Québec building set. It uses `configs/base.yaml` by default (which composes training, SB3 policy, reward, and building selection).

#### Run a training job

```bash
python -m scripts.main
```

#### Common overrides

- **Select the SB3 algorithm config**:

```bash
python -m scripts.main policy=ppo
python -m scripts.main policy=sac
python -m scripts.main policy=dqn
```

- **Control number of parallel envs**:

```bash
python -m scripts.main training.num_train_envs=4
```

- **Set episode length / env settings**:

```bash
python -m scripts.main env.max_steps=672 env.normalize_obs=true
```

Training outputs (models, logs, tensorboard, test rollouts) are written under the Hydra run directory.

---

### Multizones reference buildings

The **multizones_reference_buildings** dataset contains 6000 parametrically varied EnergyPlus buildings (1000 per type) across six ASHRAE 90.1-2022 prototypes:

- `Warehouse`, `HotelSmall`, `RetailStandalone`, `RestaurantFastFood`, `OfficeMedium`, `OfficeSmall`

Pre-generated **train/test splits** (900 / 100 per type) are stored as pickle files under `b2b/sources/data/` (e.g. `OfficeSmall_train_data`, `OfficeSmall_test_data`).

Two HVAC system types are present:
- **Unitary systems** (Warehouse, HotelSmall, RetailStandalone, RestaurantFastFood, OfficeSmall) — each zone has an independent unitary HVAC unit.
- **VAV air-loop systems** (OfficeMedium) — zones share air loops with variable air volume terminals. Cooling setpoints are excluded from the agent action space and fixed internally.

#### RL training on multizones: `scripts/train_multizones.py`

Train an SB3 agent on a **single building** selected by its position in a split list. Uses `configs/train_multizones.yaml`.

```bash
# Train PPO on the 1st building of the OfficeSmall train split (default)
PYTHONPATH=. python scripts/train_multizones.py

# Train on the 3rd building of the OfficeMedium train split
PYTHONPATH=. python scripts/train_multizones.py \
    multizones.building_type=OfficeMedium multizones.split=train multizones.index=2

# Use SAC instead of PPO
PYTHONPATH=. python scripts/train_multizones.py policy=sac

# Quick debugging run (10 days, 10k steps)
PYTHONPATH=. python scripts/train_multizones.py \
    training.total_timesteps=10000 env.max_steps=960
```

The `multizones.index` parameter is **0-based**: index 0 is the first building in the split list, index 2 is the third, etc.

Training outputs (models, logs, tensorboard) are written under `outputs/train_multizones/<building_type>/<split>_<index>/<algorithm>/<timestamp>/`.

#### Baseline rollouts on multizones: `scripts/rollout_multizones.py`

Evaluate a policy across multiple buildings and types. Uses `configs/rollout_multizones.yaml`.

```bash
# Run the default baseline on all 6 types (5 buildings each)
PYTHONPATH=. python scripts/rollout_multizones.py

# Specific types and count
PYTHONPATH=. python scripts/rollout_multizones.py \
    'multizones.types=[OfficeSmall,Warehouse]' multizones.n_per_type=3

# Different baseline controller
PYTHONPATH=. python scripts/rollout_multizones.py policy=unitary_sat

# Evaluate a trained SB3 checkpoint
PYTHONPATH=. python scripts/rollout_multizones.py \
    policy=sb3 policy.algorithm=ppo policy.checkpoint_path=/path/to/best_model.zip

# Evaluate an arbitrary Python policy class
PYTHONPATH=. python scripts/rollout_multizones.py \
    policy=custom policy.module=my_package.policies policy.class_name=MyPolicy
```

---

### Repository structure (detailed)

#### Top-level

- **`b2b/`**: main Python package
- **`configs/`**: Hydra configuration tree (policies, rewards, buildings, training, benchmarks)
- **`scripts/`**: CLI entrypoints (Hydra apps)
- **`tests/`**: pytest suite + fixtures
- **`images/`**: documentation images
- **`pyproject.toml`**: packaging + dependencies

#### `b2b/` package

- **`b2b/make_env.py`**
  - Central **environment factory**: `make_env(config, eplus_output_dir)`
  - Handles deterministic Hydro-Québec selection via `bldg.selection`
  - Creates a UUID subfolder under the EnergyPlus output directory

- **`b2b/simulator/`**
  - Gymnasium environment creation (`create_simulator`)
  - Observation/action space definitions (`observation_spaces.py`, `action_spaces.py`)
  - Rewards (`rewards.py`)
  - Wrappers/utilities (`wrappers.py`, `transform_utils.py`)

- **`b2b/pipeline/`**
  - Building preparation pipeline (parsing EDD/reports, controllable actuator discovery, schedule/surface processing, simulation steps)

- **`b2b/sources/`**
  - Dataset connectors and selectors (Hydro-Québec, NREL, OneClimate, multizones reference buildings)
  - `multizones_reference_buildings.py`: 6000-building dataset (6 types × 1000)
  - Pre-generated train/test splits (900/100) stored as pickle lists under `b2b/sources/data/`

- **`b2b/baselines/`**
  - **Controllers** live in `b2b/baselines/controllers/`
    - Rule-based policies expose SB3-like `predict()` and optionally `bind_env/reset/step_metrics`
    - `unitary_sat.py` handles buildings with one or more unitary HVAC systems
    - `air_loop_sat.py` handles VAV air-loop buildings (e.g. OfficeMedium)
  - **SB3 trainers**: `online_trainer.py` (Hydro-Québec), `multizones_trainer.py` (multizones reference buildings)
  - **SB3 interaction utilities** (callbacks, evaluation helpers)
  - `b2b/baselines/runner.py` is a thin compatibility wrapper; the rollout executor is centralized in `b2b/benchmark/`

- **`b2b/benchmark/`**
  - Centralized **simulation execution** (`runner.py`) with Gymnasium-standard loops
  - `rollout_multizones.py`: multizones rollout driver supporting baselines, SB3 checkpoints (`policy=sb3`), and custom policies (`policy=custom`)
  - Problem definitions (e.g. `problem_adaptive_dynamics.py`)
  - Experiment/Hydra wrappers under `b2b/benchmark/experiments/`
  - Baseline rollout implementation used by `scripts/baselines.py` (`baseline_rollout.py`)

- **`b2b/env.py`**, **`b2b/store.py`**
  - Download/caching utilities (including EnergyPlus binaries)

---

### Notes / troubleshooting

- **Hydra run dirs**: Hydra changes the working directory into a run folder unless configured otherwise. This is why scripts often write outputs relative to `Path.cwd()`.
- **Slow tests**: some integration tests that download/process building datasets are intentionally slow; prefer running targeted tests during development.

