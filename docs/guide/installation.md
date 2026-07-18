# Installation

## System Requirements

- Python 3.10 or later
- EnergyPlus 24.1 (downloaded automatically or installed manually)
- Linux or macOS (Windows is not tested)

## Install the Package

```bash
git clone https://github.com/vtaboga/building2building.git && cd building2building
python -m venv .venv
source .venv/bin/activate
```

### Install Variants

| Command | What it installs |
|---|---|
| `pip install -e .` | Core package only (environments, data, benchmarks) |
| `pip install -e ".[training]"` | + PyTorch, SB3, Hydra, Optuna, W&B, TensorBoard, matplotlib |
| `pip install -e ".[test]"` | + pytest, deepdiff, pytest-mock, psutil |
| `pip install -e ".[test,training]"` | Recommended dev install (the quick test suite imports `baselines`, which needs the training extras) |
| `pip install -e ".[dev]"` | + black, pyright |
| `pip install -e ".[docs]"` | + mkdocs-material, mkdocstrings |
| `pip install -e ".[all]"` | Everything above |

### Core Dependencies

| Package | Purpose |
|---|---|
| `minergym` | EnergyPlus simulation backend |
| `gymnasium` | RL environment interface |
| `numpy` | Numerical arrays |
| `huggingface-hub` | Dataset download |
| `duckdb` / `pyarrow` | Building registry queries |
| `cattrs` | Structured deserialization |
| `rdflib` | Ontology-based equipment discovery |

### Training Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network training |
| `stable-baselines3` / `sb3-contrib` | RL algorithm implementations |
| `hydra-core` / `omegaconf` | Configuration management |
| `optuna` | Hyperparameter tuning |
| `wandb` | Experiment tracking |

## EnergyPlus Setup

### Option 1: Automatic

EnergyPlus 24.1 is downloaded and cached automatically via the `minergym`
dependency; `setup_energyplus_path()` runs when `building2building` is
imported. No action needed.

### Option 2: Manual Install

Download EnergyPlus 24.1 from
[energyplus.net](https://energyplus.net/downloads) and set the path:

```bash
export ENERGYPLUS_PATH=/usr/local/EnergyPlus-24-1-0
```

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `ENERGYPLUS_PATH` | Path to an existing EnergyPlus installation | Auto-downloaded |
| `STORE_PATH` | Cache for the auto-downloaded EnergyPlus and the offline dataset-generation store | `$SCRATCH/b2b` if set, else `~/.cache/b2b` |

The pre-processed building dataset itself is downloaded from HuggingFace into
`~/.cache/building2building` (this location is fixed, not controlled by
`STORE_PATH`).

## Baselines Dependencies

The `baselines/` scripts require the training dependencies. Install them via
the main package extras:

```bash
pip install -e ".[training]"
```

## Verification

### Quick Smoke Test

```bash
python -c "import building2building as b2b; print(b2b.list_building_types())"
```

Expected output:

```
['SingleFamilyHouse', 'Warehouse', 'RetailStandalone', 'RestaurantFastFood', 'OfficeMedium', 'OfficeSmall']
```

### Run the Test Suite

```bash
pytest -m quick                           # Fast tests (no EnergyPlus simulation)
B2B_RUN_LONG_TESTS=1 pytest -m long      # Simulation-heavy tests
B2B_RUN_LONG_TESTS=1 pytest              # Full suite
```

## Troubleshooting

**`ModuleNotFoundError: minergym`** -- minergym is installed from a Git
dependency. Run `pip install -e .` again to ensure it is fetched.

**`FileNotFoundError: EnergyPlus not found`** -- set `ENERGYPLUS_PATH` to point
to your EnergyPlus installation directory.

**Download errors** -- B2B downloads building data from HuggingFace on first
use. Ensure network access is available, or pre-download with:

```python
from building2building.data.download import download_building_type
download_building_type("OfficeSmall")
```
