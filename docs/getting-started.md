# Getting Started

This guide walks you through installing B2B, creating your first environment,
running a baseline controller, and training an RL agent.

---

## 1. Installation

Create a virtual environment and install B2B in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

B2B requires **EnergyPlus 24.1**. Set the path:

```bash
export ENERGYPLUS_PATH=/usr/local/EnergyPlus-24-1-0
```

!!! info "Detailed installation"

    See [Installation](guide/installation.md) for complete system requirements,
    optional extras, and troubleshooting.

Verify the installation:

```bash
python -c "import building2building as b2b; print(b2b.list_building_types())"
pytest -m quick
```

---

## 2. Creating Your First Environment

The primary entry point is `b2b.make_env()`:

```python
import building2building as b2b

# List available building types
print(b2b.list_building_types())
# ['SingleFamilyHouse', 'OfficeSmall', 'OfficeMedium', ...]

# List building IDs in a split
train_ids = b2b.list_buildings("OfficeSmall", split="train")
print(f"{len(train_ids)} training buildings available")

# Create an environment
env = b2b.make_env(
    "OfficeSmall",
    split="train",
    index=0,
    task="task_const_e0",
)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

Key parameters of `make_env`:

| Parameter | Description | Default |
|---|---|---|
| `building_type` | One of the 6 building types | (required) |
| `split` | `"train"` or `"test"` | `"train"` |
| `index` | Position in the split list | `0` |
| `building_id` | Explicit building ID (alternative to split+index) | `None` |
| `task` | preset name (e.g. `"task_const_e0"`) or a `TaskPreset` | `"task_const_e0"` |
| `run_period` | `"full_year"`, `"winter"`, or `"summer"` | `"full_year"` |
| `timesteps_per_hour` | Simulation resolution | `12` (5-min steps) |

---

## 3. Environment Metadata

Each environment exposes rich metadata:

```python
env = b2b.make_env("OfficeSmall", task="task_const_e0")

obs_names = env.metadata["observation_names"]  # list[str]
act_names = env.metadata["action_names"]       # list[str]
equipment = env.metadata["hvac_equipment"]     # list[Equipment]
morphology = env.metadata["morphology"]        # Morphology graph

print(f"Obs dim: {env.observation_space.shape[0]}")
print(f"Act dim: {env.action_space.shape[0]}")
print(f"Obs names: {obs_names[:5]}...")
print(f"Action names: {act_names}")
env.close()
```

---

## 4. Running a Random Agent

```python
import building2building as b2b

env = b2b.make_env("OfficeSmall", split="train", index=0, task="task_const_e0")

obs, info = env.reset()
total_reward = 0.0
steps = 0
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    done = terminated or truncated

print(f"Episode finished after {steps} steps with return {total_reward:.2f}")
env.close()
```

---

## 5. Running a Baseline Controller

B2B ships reactive controllers in the `baselines/` directory. Run them via Hydra:

```bash
# Evaluate the reactive controller on one building
python -m baselines.run_reactive_control experiment=eval_reactive_control \
    building_types=[OfficeSmall] tasks=[task_const_e0] max_buildings_per_type=1
```

You can also use the controllers programmatically:

```python
import building2building as b2b
from baselines.controllers import UnitaryHvacConfig, UnitaryHvacPolicy
from baselines.utils.evaluation import run_episode

env = b2b.make_env("OfficeSmall", task="task_const_e0")
policy = UnitaryHvacPolicy(UnitaryHvacConfig())
policy.bind_env(env)

result = run_episode(env, policy)
print(f"Episode return: {result.total_reward:.1f}")
env.close()
```

---

## 6. Training with PPO

=== "Hydra CLI"

    ```bash
    python -m baselines.train_ppo experiment=train_ppo \
        building_types=[OfficeSmall] tasks=[task_const_e0] \
        buildings_per_type=1 training.total_timesteps=100000
    ```

=== "Python Script"

    ```python
    from stable_baselines3 import PPO
    import building2building as b2b

    env = b2b.make_env(
        "OfficeSmall",
        split="train",
        index=0,
        task="task_const_e0",
        run_period="winter",
    )
    env = b2b.NormalizeObservation(env)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=100_000)
    model.save("ppo_office_small")
    env.close()
    ```

---

## 7. Using a Benchmark

The benchmarks API provides structured train/test splits:

```python
import building2building as b2b

bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task_const_e0")
print(f"Train buildings: {len(bench.train_building_ids())}")
print(f"Test buildings: {len(bench.test_building_ids())}")

# Create environments
train_envs = bench.make_train_envs(n=2)
test_envs = bench.make_test_envs(n=2)

# ... train and evaluate ...

for env in train_envs + test_envs:
    env.close()
```

---

## 8. Normalized Scoring

Use the packaged baseline returns to compute normalized scores:

```python
import building2building as b2b

score = b2b.compute_normalized_score(
    cumulative_return=-5000.0,
    building_type="OfficeSmall",
    task="task_const_e0",
    run_period="full_year",
    building_id="OfficeSmall-0001",
)
print(f"Normalized score: {score:.3f}")
# > 1.0 means the agent outperforms the reactive baseline
```

---

## Next Steps

- Learn about the [environment architecture](guide/environments.md)
- Explore the [6,000 buildings](guide/buildings.md) in the dataset
- Understand the [morphology graph](guide/morphology.md) for structured representations
- Configure experiments with [Hydra](guide/configuration.md)
- Follow the [tutorials](tutorials/quick-tour.md) for hands-on examples
