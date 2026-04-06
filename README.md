## Building2Building

Building2Building is a benchmark suite for reinforcement learning in building energy management, built on top of Gymnasium and EnergyPlus.

It provides:

- **6 building types** across residential and commercial categories
- **4 benchmark problems** testing different generalization axes
- **Pre-processed buildings** downloadable from HuggingFace
- **Named task presets** reproducing exact paper conditions
- **Normalized scoring** against reactive-controller baselines

---

## Quick Start

### Installation

```bash
pip install -e .                       # Core (environments only)
pip install -e ".[training]"           # + PyTorch, SB3, wandb, etc.
pip install -e ".[training,test,dev]"  # Everything
```

### Usage

```python
import gymnasium as gym
import building2building as b2b

# One-liner: get a registered Gymnasium environment
env = gym.make("b2b/OfficeSmall-v0", split="train", index=0, task="task1")

# Or use the functional API for full control
env = b2b.new_make_env(
    building_type="OfficeSmall",
    split="train",
    index=0,
    reward=b2b.DeadbandRewardConfig(energy_weight=0.01, dT=1.0),
    run_period="winter",
)

# Get a benchmark problem
benchmark = b2b.benchmarks.DynamicsAdaptation(difficulty="easy")
train_envs = benchmark.make_train_envs(n=8)
test_envs = benchmark.make_test_envs(n=8)

# List what's available
b2b.list_building_types()  # ["SingleFamilyHouse", "OfficeSmall", ...]

# Normalized scoring
score = b2b.compute_normalized_score(-50000.0, "OfficeSmall", "task1")
```

---

## Benchmark Problems

Each benchmark class corresponds to a generalization axis from the paper:

| Class | What varies | What stays fixed |
|-------|------------|------------------|
| `GoalAdaptation` | Reward / task | Building, action space |
| `DynamicsAdaptation` | Building dynamics | Reward, action space |
| `ActionSpaceTransfer` | Controllable actuators | Building, reward |
| `CrossDomainGeneralization` | Building type | Reward, action space |

Example:

```python
from building2building.benchmarks import DynamicsAdaptation

bm = DynamicsAdaptation(difficulty="easy", task="task1")
train_envs = bm.make_train_envs(n=4)
test_envs = bm.make_test_envs(n=4)
```

---

## Task Presets

The paper defines four named task presets:

| Task | Reward | Energy Weight | Temperature Mode | dT |
|------|--------|--------------|------------------|----|
| `task1` | Deadband | 0.01 | Constant | 1.0 |
| `task2` | Deadband | 0.10 | Constant | 1.0 |
| `task3` | Deadband | 0.01 | Occupancy | 1.0 |
| `task4` | Barrier | 0.01 | Constant | 1.0 |

---

## Building Types

| Type | Zones | HVAC | Description |
|------|-------|------|-------------|
| `SingleFamilyHouse` | 1 | Unitary | Residential single-zone houses |
| `OfficeSmall` | 5 | Unitary | Small office prototype |
| `OfficeMedium` | 15+ | Central | Medium office prototype |
| `RetailStandalone` | 4 | Unitary | Standalone retail store |
| `RestaurantFastFood` | 2 | Unitary | Fast food restaurant |
| `Warehouse` | 3 | Unitary | Warehouse prototype |

---

## Legacy API

The typed `EnvBuildConfig`-based API and Hydra bridge remain available for
existing training scripts:

```python
from building2building.api import make_env, make_env_from_hydra_config
from building2building.api import make_single_zone_env, make_multizones_env
```

Training scripts in `scripts/training/` use the Hydra bridge via
`make_env_from_hydra_config`.

---

## Training Entry Points

### Single-zone houses

```bash
python scripts/training/train_single_zone_houses.py
python scripts/training/train_single_zone_houses.py policy=ppo
```

### Multizones single-building

```bash
python scripts/training/train_multizones.py
python scripts/training/train_multizones.py bldg.building_type=OfficeMedium
```

### Baselines

```bash
python scripts/training/baselines.py
python scripts/training/baselines.py policy=unitary_g36
```

---

## Tests

```bash
pytest -m quick                          # Fast tests (no EnergyPlus)
B2B_RUN_LONG_TESTS=1 pytest -m long     # Simulation-heavy tests
B2B_RUN_LONG_TESTS=1 pytest             # Full suite
```

---

## Coding Standard

- Formatter: `black` (line length 88, target py310)
- Type hints required on all functions
- Use `pathlib.Path` for all paths

```bash
python3 -m black building2building scripts tests
```

---

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve     # Live preview at http://localhost:8000
```

---

## EnergyPlus Setup

B2B can download and cache EnergyPlus automatically. Optional environment
variables:

- `ENERGYPLUS_PATH`: use an existing local EnergyPlus install
- `STORE_PATH`: choose cache/download location
