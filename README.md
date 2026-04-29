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
env = b2b.new_make_env("OfficeSmall", split="train", index=0, task="task1")

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

The paper defines five named task presets that control the reward function and
target temperature behaviour:

| Task | Reward | Energy Weight | Temperature Mode | dT |
|---|---|---|---|---|
| `task1` | Deadband | 0.01 | Constant | 1.0 |
| `task2` | Deadband | 0.10 | Constant | 1.0 |
| `task3` | Deadband | 0.01 | Occupancy (seasonal unoccupied setpoint) | 1.0 |
| `task4` | Barrier | 0.01 | Constant | 1.0 |
| `task5` | Deadband | 0.01 | Random daily schedule (per-building-type distribution) | 1.0 |

```python
env = b2b.new_make_env("OfficeSmall", task="task2")
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
bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task1")
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
    task="task1",
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

---

## Tests

```bash
pytest -m quick                           # Fast tests (no EnergyPlus)
B2B_RUN_LONG_TESTS=1 pytest -m long      # Simulation-heavy tests
B2B_RUN_LONG_TESTS=1 pytest              # Full suite
```

---

## Known Issues and Limitations

### Confirmed Bugs

- **`compute_normalized_score` argument order swapped** in
  `baselines/eval_ppo.py` and `baselines/eval_dynamics_adaptation.py`: the first
  two positional arguments (cumulative return and building type) are reversed.
- **`eval_ppo.py` model path mismatch**: `train_ppo.py` saves models in
  nested directories (`models/<type>/<task>/ppo_<id>.zip`) but `eval_ppo.py`
  expects flat filenames (`ppo_<type>_<id>_<task>.zip`).
- **`plot_ppo_specialist.py` CSV schema mismatch**: expects a `reward_mean`
  column but `eval_ppo.py` outputs `reward`.

### Missing Dependencies

`baselines/requirements.txt` is incomplete. It does not list `hydra-core`,
`omegaconf`, `optuna`, `matplotlib`, or `pyyaml`. These are covered by
`pip install -e ".[training]"` from the main package.

### Other Limitations

- `eval_dynamics_adaptation.py` hard-codes `PadObservation(env, target_size=20)`
  which may not match the value computed during training.
- No tests exist for `baselines/` code (controllers, training, evaluation).
- License is not yet selected.

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
