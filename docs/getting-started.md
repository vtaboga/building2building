# Getting Started

This guide walks you through installing B2B, creating your first environment, running a baseline controller, and training an RL agent.

---

## 1. Installation

Create a virtual environment and install B2B in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

B2B requires **EnergyPlus 24.1** to be installed on your system. Set the path:

```bash
export ENERGYPLUS_PATH=/usr/local/EnergyPlus-24-1-0
```

!!! info "Detailed installation"

    See [Installation](guide/installation.md) for complete system requirements, optional extras, and troubleshooting.

Verify the installation:

```bash
pytest -m quick
```

---

## 2. Creating Your First Environment

### Single-Zone House

The simplest way to get started is with a single-zone house:

```python
from b2b.api import make_single_zone_env

env = make_single_zone_env(
    split="train",
    split_index=0,
    eplus_output_dir="outputs/eplus",
    task={"run_period": "winter"},
    reward={"reward_type": "BarrierRewardConfig"},
)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

### Multi-Zone Reference Building

For multi-zone buildings from the ASHRAE 90.1 prototypes:

```python
from b2b.api import make_multizones_env

env = make_multizones_env(
    building_type="OfficeSmall",
    split="train",
    split_index=0,
    eplus_output_dir="outputs/eplus",
    task={"run_period": "winter"},
    reward={"reward_type": "DeadbandRewardConfig", "dT": 1.0, "energy_weight": 0.001},
)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

!!! note "Building types"

    Available building types for `make_multizones_env`: `Warehouse`, `HotelSmall`, `RetailStandalone`, `RestaurantFastFood`, `OfficeMedium`, `OfficeSmall`.

### Full Config API

For maximum control, use `EnvBuildConfig` directly:

```python
from b2b.api import make_env
from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig
from b2b.types import TaskConfig, reward_config_from_dict

cfg = EnvBuildConfig(
    dataset_selection=DatasetSelectionConfig(
        dataset="multizones_reference_buildings",
        building_type="OfficeMedium",
        split="train",
        mode="split_index",
        split_index=5,
    ),
    task=TaskConfig.from_dict({
        "run_period": "summer",
        "target_temperature_mode": "occupancy",
        "default_zone_target_temperature": {
            "occupied_c": 23.0,
            "unoccupied_c": 28.0,
        },
    }),
    reward=reward_config_from_dict({
        "reward_type": "BarrierRewardConfig",
        "energy_weight": 0.1,
        "deadband_c": 0.5,
        "violation_penalty": 100.0,
    }),
    env_max_steps=8640,
)
env = make_env(cfg, eplus_output_dir="outputs/eplus")
```

---

## 3. Running a Random Agent

```python
from b2b.api import make_single_zone_env

env = make_single_zone_env(
    split="train",
    split_index=0,
    eplus_output_dir="outputs/eplus",
)

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

print(f"Episode finished after {steps} steps with total reward {total_reward:.2f}")
env.close()
```

---

## 4. Running a Baseline Controller

B2B includes several hand-crafted baseline controllers. You can run them via Hydra:

```bash
python -m b2b.benchmark.baseline_rollout policy=fan_coil_constant \
    bldg=single_family \
    task.run_period=winter \
    reward=barrier
```

Baseline controllers follow the same `predict(obs, deterministic) -> (action, state)` interface as SB3 policies, so they integrate seamlessly with the rollout infrastructure.

---

## 5. Training with PPO

=== "Hydra CLI"

    ```bash
    python -m b2b.train \
        policy=ppo \
        bldg=single_family \
        task.run_period=winter \
        reward=deadband \
        training.total_timesteps=500000 \
        wandb.enabled=true
    ```

=== "Python Script"

    ```python
    from stable_baselines3 import PPO
    from b2b.api import make_single_zone_env
    from b2b.simulator.wrappers import NormalizeObservation

    env = make_single_zone_env(
        split="train",
        split_index=0,
        eplus_output_dir="outputs/eplus",
        task={"run_period": "winter"},
        reward={"reward_type": "DeadbandRewardConfig", "dT": 1.0, "energy_weight": 0.001},
    )
    env = NormalizeObservation(env)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=500_000)
    model.save("ppo_single_zone")
    env.close()
    ```

---

## 6. Training with SAC

=== "Hydra CLI"

    ```bash
    python -m b2b.train \
        policy=sac \
        bldg=single_family \
        task.run_period=winter \
        reward=barrier \
        training.total_timesteps=500000
    ```

=== "Python Script"

    ```python
    from stable_baselines3 import SAC
    from b2b.api import make_single_zone_env
    from b2b.simulator.wrappers import NormalizeObservation

    env = make_single_zone_env(
        split="train",
        split_index=0,
        eplus_output_dir="outputs/eplus",
        task={"run_period": "winter"},
        reward={"reward_type": "BarrierRewardConfig", "energy_weight": 0.1},
    )
    env = NormalizeObservation(env)

    model = SAC("MlpPolicy", env, verbose=1, buffer_size=100_000, batch_size=128)
    model.learn(total_timesteps=500_000)
    model.save("sac_single_zone")
    env.close()
    ```

---

## 7. Evaluating on a Benchmark Split

After training, evaluate your policy on the held-out test split:

```python
from stable_baselines3 import PPO
from b2b.api import make_single_zone_env
from b2b.simulator.wrappers import NormalizeObservation
import numpy as np

model = PPO.load("ppo_single_zone")

returns = []
for idx in range(10):
    env = make_single_zone_env(
        split="test",
        split_index=idx,
        eplus_output_dir=f"outputs/eval/{idx}",
        task={"run_period": "winter"},
        reward={"reward_type": "DeadbandRewardConfig", "dT": 1.0, "energy_weight": 0.001},
    )
    env = NormalizeObservation(env)

    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    returns.append(total_reward)
    env.close()

print(f"Mean return over 10 test buildings: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
```

---

## Next Steps

- Learn about the [environment architecture](guide/environments.md)
- Explore the [7,000+ buildings](guide/buildings.md) in the dataset
- Understand the [reward functions](guide/rewards.md) available
- Configure experiments with [Hydra](guide/configuration.md)
- Write [custom policies](guide/custom-policies.md)
