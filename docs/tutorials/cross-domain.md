# Cross-Domain with Amorpheus

This tutorial shows how the Amorpheus type-heterogeneous transformer handles
cross-domain transfer between different building types.

A standalone script is available at `tutorials/run_cross_domain.py`.

---

## Overview

Cross-domain transfer trains on one building type and tests on a different
type. The Amorpheus architecture uses the morphology graph to apply
per-node-type encoders/decoders, enabling a single policy to work across
building types with different observation and action dimensions.

## Step 1: Inspect the Benchmark

```python
import building2building as b2b

bench = b2b.benchmarks.CrossDomainGeneralization(difficulty="easy", task="task_const_e0")
print(f"Train type: {bench.train_type}")   # RetailStandalone
print(f"Test type: {bench.test_type}")     # OfficeSmall

# Compare the two building types
train_env = b2b.new_make_env(bench.train_type, split="train", index=0, task="task_const_e0")
test_env = b2b.new_make_env(bench.test_type, split="test", index=0, task="task_const_e0")

print(f"\nTrain: obs={train_env.observation_space.shape[0]}, act={train_env.action_space.shape[0]}")
print(f"Test:  obs={test_env.observation_space.shape[0]}, act={test_env.action_space.shape[0]}")
train_env.close()
test_env.close()
```

## Step 2: Explore the Morphology

```python
env = b2b.new_make_env("RetailStandalone", split="train", index=0, task="task_const_e0")
morph = env.metadata["morphology"]

print(f"Nodes: {len(morph.nodes)}")
for node in morph.nodes:
    print(f"  {node.node_id}: type={node.node_type.name}, "
          f"obs_dim={node.node_type.observation_dim}, "
          f"act_dim={node.node_type.action_dim}")
env.close()
```

## Step 3: Instantiate Amorpheus

```python
import torch
from baselines.models.amorpheus import AmorpheusPolicy

env = b2b.new_make_env("RetailStandalone", split="train", index=0, task="task_const_e0")
morph = env.metadata["morphology"]

policy = AmorpheusPolicy(morphology=morph, embed_dim=64)
n_params = sum(p.numel() for p in policy.parameters())
print(f"Amorpheus parameters: {n_params:,}")

# Forward pass
obs_t = torch.randn(1, env.observation_space.shape[0])
actions, values = policy(obs_t)
print(f"Actions shape: {actions.shape}")
print(f"Values shape: {values.shape}")
env.close()
```

## Step 4: Transfer to a Different Building Type

```python
# Switch to OfficeSmall
env2 = b2b.new_make_env("OfficeSmall", split="test", index=0, task="task_const_e0")
morph2 = env2.metadata["morphology"]

# The SAME policy parameters, different morphology
policy.morphology = morph2
obs_t2 = torch.randn(1, env2.observation_space.shape[0])
actions2, values2 = policy(obs_t2)
print(f"OfficeSmall actions shape: {actions2.shape}")
env2.close()
```

## Step 5: Train with the CLI

For a full training run:

```bash
python -m baselines.train_cross_domain experiment=train_cross_domain

# Short version for testing
python -m baselines.train_cross_domain experiment=train_cross_domain \
    total_timesteps=10000
```

## Step 6: Evaluate

```bash
python -m baselines.eval_cross_domain \
    --model-path outputs/.../amorpheus_policy.pt \
    --test-building-types Warehouse SingleFamilyHouse
```
