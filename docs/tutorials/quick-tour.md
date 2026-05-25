# Quick Tour

This tutorial walks you through the core B2B API in 10 minutes: creating
environments, inspecting spaces, running an episode, and exploring the
morphology graph.

A standalone script is available at `tutorials/quick_tour.py`.

---

## Step 1: Import and List Building Types

```python
import building2building as b2b

types = b2b.list_building_types()
print(f"Available building types ({len(types)}): {types}")
```

Expected output:

```
Available building types (6): ['SingleFamilyHouse', 'Warehouse', 'RetailStandalone', 'RestaurantFastFood', 'OfficeMedium', 'OfficeSmall']
```

## Step 2: Create an Environment

```python
env = b2b.new_make_env("OfficeSmall", split="test", index=0, task="task_const_e0")

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Obs dim: {env.observation_space.shape[0]}")
print(f"Act dim: {env.action_space.shape[0]}")
```

## Step 3: Inspect Metadata

```python
obs_names = env.metadata["observation_names"]
act_names = env.metadata["action_names"]
equipment = env.metadata["hvac_equipment"]

print(f"\nObservation names ({len(obs_names)}):")
for i, name in enumerate(obs_names):
    print(f"  [{i}] {name}")

print(f"\nAction names ({len(act_names)}):")
for i, name in enumerate(act_names):
    print(f"  [{i}] {name}")

print(f"\nHVAC equipment ({len(equipment)} systems):")
for eq in equipment:
    print(f"  {type(eq).__name__}: {eq.zone_name}")
```

## Step 4: Run a Random-Action Episode

```python
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

    if steps <= 3:
        print(f"Step {steps}: reward={reward:.2f}")

print(f"\nEpisode finished: {steps} steps, return={total_reward:.2f}")
```

## Step 5: Explore the Morphology Graph

```python
import numpy as np

env2 = b2b.new_make_env("OfficeSmall", split="test", index=0, task="task_const_e0")
morph = env2.metadata["morphology"]

print(f"Nodes: {len(morph.nodes)}")
print(f"Edges: {len(morph.edges)}")
print(f"Type counts: {morph.type_counts()}")

obs, _ = env2.reset()
local_obs = morph.split_observation(obs)
print("\nPer-node local observations:")
for node_id, local in local_obs.items():
    print(f"  {node_id}: shape={local.shape}, values={local}")
```

## Step 6: Compare Building Types

```python
for btype in ["SingleFamilyHouse", "OfficeSmall", "OfficeMedium"]:
    env_tmp = b2b.new_make_env(btype, split="test", index=0, task="task_const_e0")
    morph_tmp = env_tmp.metadata["morphology"]
    print(f"{btype}: obs={env_tmp.observation_space.shape[0]}, "
          f"act={env_tmp.action_space.shape[0]}, "
          f"nodes={len(morph_tmp.nodes)}, "
          f"types={morph_tmp.type_counts()}")
    env_tmp.close()
```

## Step 7: Clean Up

```python
env.close()
env2.close()
```
