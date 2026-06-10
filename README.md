# Building2Building

Building2Building (B2B) is a large-scale reinforcement learning benchmark for
HVAC control in buildings. It exposes over 7,000 parametrically generated
EnergyPlus building models as Gymnasium environments, spanning 6 building types,
16 ASHRAE climate zones, and 3 HVAC system types.

B2B is designed to accelerate research in **transfer learning**, **multi-task
RL**, and **meta-learning** for building energy management.

**Key features:**

- 7,000+ buildings from ASHRAE 90.1-2022 prototypes and residential archetypes
- 6 building types: `SingleFamilyHouse`, `OfficeSmall`, `OfficeMedium`, `RetailStandalone`, `RestaurantFastFood`, `Warehouse`
- 4 named task presets reproducing exact paper conditions
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

EnergyPlus 24.1 is required. Set the path if it is not on your system `PATH`:

```bash
export ENERGYPLUS_PATH=/path/to/EnergyPlus-24-1-0
```

Building data is hosted on HuggingFace (`vtaboga/building2building_dataset`) and
downloaded automatically on first use.

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

## Building Types

| Type | Zones | HVAC System | Description |
|---|---|---|---|
| `SingleFamilyHouse` | 1 | Unitary | Residential single-zone houses |
| `OfficeSmall` | 5 | Unitary | Small office prototype |
| `OfficeMedium` | 15+ | VAV air-loop | Medium office prototype |
| `RetailStandalone` | 4 | Unitary | Standalone retail store |
| `RestaurantFastFood` | 2 | Unitary | Fast food restaurant |
| `Warehouse` | 3 | Unitary | Warehouse prototype |

---

## Task Presets

Nine named presets form a 3×3 grid over setpoint mode and energy weight,
all using the normalized deadband reward with per-bucket `(tau_T, tau_E)`
calibration constants:

| Task | Mode | Energy Weight |
|---|---|---|
| `task_const_e0` | Constant | 0.0 (comfort-only) |
| `task_const_emed` | Constant | 1.0 (balanced) |
| `task_const_ehigh` | Constant | 5.0 (energy-emphasis) |
| `task_occ_e0` | Occupancy (seasonal) | 0.0 |
| `task_occ_emed` | Occupancy (seasonal) | 1.0 |
| `task_occ_ehigh` | Occupancy (seasonal) | 5.0 |
| `task_rand_e0` | Random schedule | 0.0 |
| `task_rand_emed` | Random schedule | 1.0 |
| `task_rand_ehigh` | Random schedule | 5.0 |

The default is `task_const_e0` (constant setpoint, comfort-only).

```python
env = b2b.make_env("OfficeSmall", task="task_occ_emed")
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
`building2building` API. They include reactive controllers, PPO training,
dynamics adaptation, cross-domain transfer with Amorpheus, and evaluation /
plotting.

```bash
# Reactive control baseline evaluation
python -m baselines.run_reactive_control experiment=eval_reactive_control

# Train per-building PPO specialists
python -m baselines.train_ppo experiment=train_ppo

# Dynamics adaptation with building-parameter augmentation
python -m baselines.train_dynamics_adaptation experiment=train_dynamics_parameterized
```

See [`baselines/README.md`](baselines/README.md) for the full reference.

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
# score > 1.0 means the agent outperforms the baseline
```

> **Note:** the packaged baseline file can be regenerated by running
> `python -m baselines.run_reactive_control experiment=eval_reactive_control`.

---

## Documentation

Full documentation is built with MkDocs Material:

```bash
pip install -e ".[docs]"
mkdocs serve          # http://localhost:8000
```

To reproduce every figure and table from the paper, see
[`REPRODUCING.md`](REPRODUCING.md).

The four benchmark problems are documented at
[`docs/benchmarks/`](docs/benchmarks/overview.md).

---

## Tests

```bash
pytest -m quick                           # Fast tests (no EnergyPlus)
B2B_RUN_LONG_TESTS=1 pytest -m long      # Simulation-heavy tests
B2B_RUN_LONG_TESTS=1 pytest              # Full suite
```

---

## Known Issues and Limitations

### Other Limitations

- No tests exist for `baselines/` code (controllers, training, evaluation).

---

## License

MIT — see [`LICENSE`](LICENSE). See [`CHANGELOG.md`](CHANGELOG.md) for
version history and [`docs/api/stability.md`](docs/api/stability.md) for the
public API contract and deprecation policy.

---

## Citation

```bibtex
@article{b2b2025,
  title   = {Building2Building: A Large-Scale Benchmark for Transfer and
             Multi-Task Reinforcement Learning in HVAC Control},
  author  = {Vincent Taboga, Justin Veilleux, Doseok Jang, Anushree Rankawat, Pierre-Luc Bacon},
  year    = {2025},
}
```
