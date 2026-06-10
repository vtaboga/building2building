# Cross-Domain Transfer Baselines

Baselines for the cross-domain generalization benchmark (Paper Section 6.2).

## Overview

The cross-domain benchmark trains the **Amorpheus** type-heterogeneous
transformer policy on multiple building types simultaneously, then evaluates
zero-shot transfer to unseen building types.

## Amorpheus Architecture

Amorpheus leverages the morphology graph to handle buildings with different
observation and action dimensions:

1. **Per-node-type encoders** map local observations to a fixed embedding
2. **Type embeddings** are added to each node
3. **Transformer encoder** aggregates information across all nodes
4. **Per-node-type decoders** produce local actions from embeddings
5. **Value head** computes state value from mean-pooled embeddings

```python
from baselines.models.amorpheus import AmorpheusPolicy

env = b2b.make_env("OfficeSmall", task="task_const_e0")
morph = env.metadata["morphology"]

policy = AmorpheusPolicy(morphology=morph, embed_dim=64)
# The SAME policy works on different building types
```

## Usage

### Training

```bash
python -m baselines.train_cross_domain experiment=train_cross_domain

# Override architecture
python -m baselines.train_cross_domain experiment=train_cross_domain \
    model.embed_dim=128 total_timesteps=5000000
```

### Evaluation

```bash
python -m baselines.eval_cross_domain \
    --model-path outputs/.../amorpheus_policy.pt \
    --test-building-types Warehouse SingleFamilyHouse
```

## Training Details

The cross-domain script uses a custom PPO training loop (not SB3) to handle
the morphology-based split/join operations:

1. Collect rollouts across multiple building types
2. For each timestep, use `morph.split_observation()` to get per-node inputs
3. Forward through Amorpheus to get per-node actions
4. Use `morph.join_actions()` to produce flat actions
5. Compute GAE advantages and PPO loss
6. Update the single shared policy

## Difficulty Levels

| Difficulty | Train Type | Test Type |
|---|---|---|
| `easy` | `RetailStandalone` | `OfficeSmall` |
| `medium` | `RetailStandalone` | `Warehouse` |
| `hard` | `OfficeSmall` | `OfficeMedium` |
