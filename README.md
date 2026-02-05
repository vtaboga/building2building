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

### RL training entrypoint: `scripts/main.py`

`scripts/main.py` is the **Stable-Baselines3 training entrypoint**. It uses `configs/base.yaml` by default (which composes training, SB3 policy, reward, and building selection).

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
  - Dataset connectors and selectors (Hydro-Québec, NREL, OneClimate, etc.)
  - Includes some stored selection lists under `b2b/sources/data/`

- **`b2b/baselines/`**
  - **Controllers** live in `b2b/baselines/controllers/`
    - Rule-based policies expose SB3-like `predict()` and optionally `bind_env/reset/step_metrics`
    - Any policy-specific “patches” should be implemented here (not in rollout loops)
  - **SB3 interaction utilities** (training, callbacks, evaluation helpers)
  - `b2b/baselines/runner.py` is a thin compatibility wrapper; the rollout executor is centralized in `b2b/benchmark/`

- **`b2b/benchmark/`**
  - Centralized **simulation execution** (`runner.py`) with Gymnasium-standard loops
  - Problem definitions (e.g. `problem_adaptive_dynamics.py`)
  - Experiment/Hydra wrappers under `b2b/benchmark/experiments/`
  - Baseline rollout implementation used by `scripts/baselines.py` (`baseline_rollout.py`)

- **`b2b/env.py`**, **`b2b/store.py`**
  - Download/caching utilities (including EnergyPlus binaries)

---

### Notes / troubleshooting

- **Hydra run dirs**: Hydra changes the working directory into a run folder unless configured otherwise. This is why scripts often write outputs relative to `Path.cwd()`.
- **Slow tests**: some integration tests that download/process building datasets are intentionally slow; prefer running targeted tests during development.

