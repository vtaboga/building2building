# Reproduce Dynamics Benchmark

This tutorial reproduces a mini version of the dynamics adaptation experiment
(Paper Section 6.1) using the parameterized approach.

A standalone script is available at `tutorials/run_dynamics_benchmark.py`.

---

## Overview

The dynamics adaptation benchmark trains a single policy across multiple
building instances of the same type, using building parameter augmentation
to help the policy generalize.

## Step 1: Generate Baseline Returns

First, generate the reactive-controller baseline for scoring:

```bash
python -m baselines.run_rule_based experiment=eval_rule_based \
    building_types=[SingleFamilyHouse] tasks=[task1] max_buildings_per_type=5
```

## Step 2: Inspect the Benchmark

```python
import building2building as b2b

bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task1")
print(f"Building type: {bench.building_type}")
print(f"Train buildings: {len(bench.train_building_ids())}")
print(f"Test buildings: {len(bench.test_building_ids())}")
```

## Step 3: Train with the CLI

The fastest way to run the experiment:

```bash
# Parameterized approach (with building parameter augmentation)
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_parameterized \
    difficulty=easy \
    training.total_timesteps=50000

# Compare with specialist (per-building PPO)
python -m baselines.train_dynamics_adaptation \
    experiment=train_dynamics_specialist \
    difficulty=easy \
    training.total_timesteps=50000
```

## Step 4: Train Programmatically

For more control, here is the Python equivalent:

```python
import building2building as b2b
from stable_baselines3 import PPO

bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task1")
train_ids = bench.train_building_ids()[:5]  # use 5 buildings for speed

# Create a multi-building training environment
def make_env(idx):
    env = b2b.new_make_env(
        bench.building_type,
        building_id=train_ids[idx % len(train_ids)],
        task="task1",
        run_period="winter",
    )
    env = b2b.PadObservation(env, target_size=20)
    env = b2b.AugmentObservationWithBuildingParams(env)
    env = b2b.NormalizeObservation(env)
    return env

env = b2b.ResampleBuildingOnResetWrapper(
    make_env,
    available_indices=list(range(len(train_ids))),
)

model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
model.learn(total_timesteps=50_000)
model.save("ppo_dynamics_easy")
env.close()
```

## Step 5: Evaluate on Test Buildings

```python
test_ids = bench.test_building_ids()[:3]
returns = []

for bid in test_ids:
    eval_env = b2b.new_make_env(
        bench.building_type,
        building_id=bid,
        task="task1",
        run_period="winter",
    )
    eval_env = b2b.PadObservation(eval_env, target_size=20)
    eval_env = b2b.AugmentObservationWithBuildingParams(eval_env)
    eval_env = b2b.NormalizeObservation(eval_env)

    obs, _ = eval_env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    returns.append(total_reward)
    eval_env.close()

import numpy as np
print(f"Test returns: {[f'{r:.0f}' for r in returns]}")
print(f"Mean: {np.mean(returns):.0f} +/- {np.std(returns):.0f}")
```

## Step 6: Plot Results

```bash
python -m baselines.plotting.plot_dynamics_adaptation \
    --specialist-csv results_specialist.csv \
    --baseline-csv results_baseline.csv \
    --parameterized-csv results_parameterized.csv
```
