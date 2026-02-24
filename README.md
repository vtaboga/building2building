## Building2Building (B2B)

Building2Building is a reinforcement learning framework for EnergyPlus building simulation, built on top of Gymnasium and `minergym`.

It provides:

- typed environment construction (`b2b.api`, `b2b.config`, `b2b.envs`)
- dataset selection across single-zone and multizone building sources
- benchmark problem orchestration
- baseline controllers and SB3 training entrypoints
- reproducible quick vs long test suites

---

## Architecture (Current)

Core package boundaries:

- `b2b.api`: public importable API for building environments
- `b2b.config`: typed config models and validation
- `b2b.datasets`: unified building selection and dataset access
- `b2b.envs`: canonical typed env factory
- `b2b.benchmark`: benchmark orchestration and problem definitions
- `b2b.baselines`: baseline policy implementations + policy registry
- `b2b.training`: training entrypoints callable from scripts or Python
- `scripts/`: thin Hydra adapters only

Design principles used in the current refactor:

- typed config objects in core internals
- Hydra/OmegaConf only at script/adaptation boundaries
- benchmark orchestration separated from policy construction
- environment creation separated from benchmark and baseline logic

---

## Quick Start

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install package

```bash
pip install -e .
```

### 3) Optional: test dependencies

```bash
pip install -e ".[test,dev]"
```

### 4) EnergyPlus setup

B2B can download/cache EnergyPlus automatically through the store layer.

Optional environment variables:

- `ENERGYPLUS_PATH`: use an existing local EnergyPlus install
- `STORE_PATH`: choose cache/download location

---

## Coding Standard

Formatting standard:

- formatter: `black`
- line length: `88`
- target version: `py310`

Run formatting:

```bash
python3 -m black b2b scripts tests
```

---

## Public Python API

```python
from pathlib import Path

from b2b.api import make_env
from b2b.config import DatasetSelectionConfig, EnvBuildConfig
from b2b.types import TaskConfig, reward_config_from_dict

cfg = EnvBuildConfig(
    dataset_selection=DatasetSelectionConfig(
        dataset="single_zone_houses",
        split="train",
        mode="split_index",
        split_index=0,
    ),
    task=TaskConfig.from_dict({"run_period": "winter"}),
    reward=reward_config_from_dict({"reward_type": "BarrierRewardConfig"}, area=1.0),
)
env = make_env(cfg, eplus_output_dir=Path("outputs/eplus"))
```

---

## Task and Reward Configuration

Task config (`configs/task/default.yaml`) supports:

- `task.run_period`: `full_year`, `winter`, `summer`
- `task.target_temperature_mode`: `constant`, `occupancy`
- `task.default_zone_target_temperature`
- `task.zone_target_temperatures` (per-zone overrides)

Reward configs:

- deadband: `configs/reward/deadband.yaml`
- barrier: `configs/reward/barrier.yaml`

Examples:

```bash
# Occupancy-driven targets + barrier reward
python scripts/train_single_zone_houses.py task.target_temperature_mode=occupancy reward=barrier

# Summer run period with deadband reward
python scripts/train_single_zone_houses.py task.run_period=summer reward=deadband
```

---

## Training Entry Points

### A) Single-zone houses training (general entrypoint)

Script: `scripts/train_single_zone_houses.py`  
Config root: `configs/base.yaml`

```bash
python scripts/train_single_zone_houses.py
python scripts/train_single_zone_houses.py policy=ppo
python scripts/train_single_zone_houses.py policy=sac
python scripts/train_single_zone_houses.py training.num_train_envs=4
python scripts/train_single_zone_houses.py env.max_steps=672
```

### B) Multizones single-building training

Script: `scripts/train_multizones.py`  
Config root: `configs/train_multizones.yaml`

```bash
python scripts/train_multizones.py
python scripts/train_multizones.py multizones.building_type=OfficeMedium multizones.split=train multizones.index=2
python scripts/train_multizones.py policy=sac
python scripts/train_multizones.py training.total_timesteps=10000 env.max_steps=960
```

---

## Benchmark Problems and Rollouts

### 1) Adaptive Dynamics Benchmark (single-zone houses)

Script: `scripts/bm_adaptive_dynamics.py`  
Config: `configs/bm_adaptive_dynamics.yaml`

```bash
# fast smoke
python scripts/bm_adaptive_dynamics.py benchmark.limit=1 benchmark.max_steps=1

# full split run
python scripts/bm_adaptive_dynamics.py benchmark.split=train benchmark.start=0 benchmark.limit=0

# change baseline policy
python scripts/bm_adaptive_dynamics.py policy=unitary_pi
python scripts/bm_adaptive_dynamics.py policy=fan_coil_constant
```

### 2) Multizones Split Benchmark Interface

Script: `scripts/benchmark_multizones_splits.py`  
Configs:

- `configs/benchmark_multizones_splits.yaml`
- `configs/benchmark_interface/single_type.yaml`
- `configs/benchmark_interface/multi_type.yaml`
- `configs/benchmark_multizones_officemedium_actuator_shift.yaml`

Single-type train/test split:

```bash
python scripts/benchmark_multizones_splits.py benchmark_interface=single_type benchmark_interface.building_type=OfficeSmall
```

Multi-type split:

```bash
python scripts/benchmark_multizones_splits.py benchmark_interface=multi_type 'benchmark_interface.train.types=[OfficeSmall,Warehouse]' 'benchmark_interface.test.types=[OfficeMedium,RetailStandalone]'
```

OfficeMedium actuator-shift benchmark:

```bash
python scripts/benchmark_multizones_splits.py --config-name benchmark_multizones_officemedium_actuator_shift
```

Reverse actuator shift direction:

```bash
python scripts/benchmark_multizones_splits.py --config-name benchmark_multizones_officemedium_actuator_shift benchmark_interface.train.config.actuator_access.include_zone_heating_setpoints=true benchmark_interface.test.config.actuator_access.include_zone_heating_setpoints=false
```

### 3) Multizones Rollout Driver

Script: `scripts/rollout_multizones.py`  
Config: `configs/rollout_multizones.yaml`

```bash
# default rollout
python scripts/rollout_multizones.py

# choose types and count
python scripts/rollout_multizones.py 'multizones.types=[OfficeSmall,Warehouse]' multizones.n_per_type=3

# evaluate SB3 checkpoint
python scripts/rollout_multizones.py policy=sb3 policy.algorithm=ppo policy.checkpoint_path=/path/to/best_model.zip

# evaluate custom policy class
python scripts/rollout_multizones.py policy=custom policy.module=my_package.policies policy.class_name=MyPolicy
```

### 4) Baseline Rollout Entry (general)

Script: `scripts/baselines.py`  
Config: `configs/baseline.yaml`

```bash
python scripts/baselines.py
python scripts/baselines.py policy=unitary_sat
python scripts/baselines.py policy=unitary_pi
python scripts/baselines.py env.max_steps=672 n_episodes=1
```

---

## Tests

Two test levels are defined:

- `quick`: no EnergyPlus simulation required
- `long`: simulation/data-heavy tests

Run quick tests:

```bash
pytest -m quick
```

Run long tests:

```bash
B2B_RUN_LONG_TESTS=1 pytest -m long
```

Run full suite:

```bash
B2B_RUN_LONG_TESTS=1 pytest
```

---

## Data/Processing Utilities

Repository data prep utilities remain available in `scripts/processing/` and `scripts/generate_dataset.py` for dataset workflows. These are utility scripts, not benchmark problem entrypoints.

---

## Notes

- Hydra run directories are enabled in script configs (`hydra.run.dir`), so output files are typically written under `outputs/...`.
- For development, prefer running quick tests continuously and long tests before merge/release.
