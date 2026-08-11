# Cross-Domain Transfer Baselines

Baselines for the cross-domain generalization benchmark.

## Overview

The cross-domain benchmark trains the **Amorpheus** type-heterogeneous
transformer policy on multiple building types simultaneously, then evaluates
zero-shot transfer to unseen building types.

## Amorpheus Architecture

Amorpheus leverages the morphology graph to handle buildings with different
observation and action dimensions:

1. **Per-node-type encoders** map local observations (with a building-type
   one-hot appended) to a fixed embedding
2. **Transformer encoder** aggregates information across all nodes
3. **Per-node-type decoders** produce per-node Beta-distribution action
   parameters from the embeddings
4. **Value head** computes the state value from mean-pooled embeddings

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
    model.n_heads=8 model.n_layers=4
```

Training-loop parameters (rollout length, number of PPO iterations,
building-pool size, ...) default to the values in the script's
`TrainConfig` dataclass and can be added from the command line, e.g.
`+training.total_iterations=400 +training.n_buildings_per_type=3`.

### Evaluation

```bash
python -m baselines.eval_cross_domain \
    --model-path outputs/.../amorpheus_policy.pt \
    --test-building-types Warehouse SingleFamilyHouse \
    --task task_occ_e0
```

Pass `--embed-dim`, `--n-heads`, and `--n-layers` matching the trained
model; `--n-test` controls how many test buildings are evaluated per type.
Results are written to `<base_dir>/b2b/eval/cross_domain_results.csv`
(override with `--output`).

## Training Details

The cross-domain script uses a custom PPO training loop (not SB3) with
Beta-distributed actions, GAE, and KL early stopping to handle the
morphology-based split/join operations:

1. Collect rollouts across multiple building types (multi-worker)
2. For each timestep, use `morph.split_observation()` to get per-node inputs
3. Forward through Amorpheus to get per-node actions
4. Use `morph.join_actions()` to produce flat actions
5. Compute GAE advantages and PPO loss
6. Update the single shared policy

## Settings

The `b2b.benchmarks.CrossDomainGeneralization` benchmark defines three
train/test building-type pairs (the training script itself takes an explicit
list of building types):

| Setting | Train Type | Test Type |
|---|---|---|
| `1` | `RetailStandalone` | `OfficeSmall` |
| `2` | `RetailStandalone` | `Warehouse` |
| `3` | `OfficeSmall` | `OfficeMedium` |
