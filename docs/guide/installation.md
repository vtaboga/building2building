# Installation

This page covers system requirements, installation steps, environment variable configuration, and verification.

---

## System Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| EnergyPlus | 24.1 |
| OS | Linux (recommended), macOS |
| RAM | 4 GB minimum, 8 GB+ recommended for multizone buildings |

---

## Installing B2B

### 1. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install in Editable Mode

=== "Core only"

    ```bash
    pip install -e .
    ```

=== "With test dependencies"

    ```bash
    pip install -e ".[test]"
    ```

=== "With dev tools"

    ```bash
    pip install -e ".[dev]"
    ```

=== "With docs"

    ```bash
    pip install -e ".[docs]"
    ```

=== "Everything"

    ```bash
    pip install -e ".[all]"
    ```

### Core Dependencies

B2B installs the following core dependencies automatically:

| Package | Purpose |
|---|---|
| `minergym` | EnergyPlus-Gymnasium bridge |
| `gymnasium` | RL environment interface |
| `stable-baselines3` | RL algorithms (PPO, SAC, DQN) |
| `sb3-contrib` | Additional algorithms (TRPO) |
| `hydra-core` | Configuration management |
| `wandb` | Experiment tracking |
| `numpy`, `pandas` | Numerical and data handling |
| `rdflib` | Ontology-based HVAC equipment discovery |
| `cattrs` | Structured serialization of equipment configs |

### Optional Dependency Groups

| Extra | Packages | Purpose |
|---|---|---|
| `test` | `pytest`, `pytest-mock`, `deepdiff` | Running the test suite |
| `dev` | `black`, `pyright` | Code formatting and type checking |
| `docs` | `mkdocs-material`, `mkdocstrings`, `pymdown-extensions` | Building documentation |
| `all` | All of the above | Full development environment |

---

## Installing EnergyPlus

B2B requires EnergyPlus 24.1 to be installed on your system.

### Linux

```bash
wget https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-SHA-Linux-Ubuntu22.04-x86_64.tar.gz
tar -xzf EnergyPlus-24.1.0-*.tar.gz
sudo mv EnergyPlus-24-1-0 /usr/local/
```

### macOS

```bash
wget https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-SHA-macOS-x86_64.tar.gz
tar -xzf EnergyPlus-24.1.0-*.tar.gz
sudo mv EnergyPlus-24-1-0 /usr/local/
```

!!! warning "Version matters"

    B2B relies on EnergyPlus 24.1 APIs and IDD schema. Other versions may cause compatibility issues with the building models and simulation outputs.

---

## Environment Variables

B2B uses two environment variables for path resolution:

### `ENERGYPLUS_PATH`

Path to the EnergyPlus installation directory. Required for running simulations.

```bash
export ENERGYPLUS_PATH=/usr/local/EnergyPlus-24-1-0
```

### `STORE_PATH`

Path to the B2B data store where datasets, intermediate build artifacts, and cached pipeline outputs are stored. Defaults to `~/.b2b_store` if not set.

```bash
export STORE_PATH=/path/to/b2b/data/store
```

!!! tip "Persisting environment variables"

    Add these exports to your `~/.bashrc` or `~/.zshrc`:

    ```bash
    echo 'export ENERGYPLUS_PATH=/usr/local/EnergyPlus-24-1-0' >> ~/.bashrc
    echo 'export STORE_PATH=/path/to/b2b/data/store' >> ~/.bashrc
    source ~/.bashrc
    ```

---

## Verifying the Installation

### Quick Tests (No Simulation)

Run the fast test suite that does not require EnergyPlus:

```bash
pytest -m quick
```

These tests validate configuration parsing, data structures, and utility functions without launching EnergyPlus.

### Full Tests (With Simulation)

Run the complete test suite including simulation-based tests:

```bash
pytest
```

!!! note "Simulation tests"

    Tests marked `long` launch EnergyPlus simulations and may take several minutes. Use `-m quick` during development for rapid iteration.

### Smoke Test

Verify that B2B can import and create an environment:

```python
from b2b.api import make_single_zone_env

env = make_single_zone_env(
    split="train",
    split_index=0,
    eplus_output_dir="/tmp/b2b_smoke_test",
)
obs, info = env.reset()
print(f"Success! Observation shape: {obs.shape}")
env.close()
```

---

## Troubleshooting

### `EnergyPlus not found`

Ensure `ENERGYPLUS_PATH` points to the directory containing the `energyplus` binary:

```bash
ls $ENERGYPLUS_PATH/energyplus
```

### `minergym` installation fails

`minergym` is installed from a Git repository. Ensure `git` is available and you have network access:

```bash
pip install "minergym @ git+https://github.com/Terramorpha/minergym.git@actuators"
```

### Import errors for `rdflib` or `cattrs`

These are pinned dependencies. Reinstall with:

```bash
pip install -e ".[all]" --force-reinstall
```
