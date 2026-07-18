# Cross-Domain with Amorpheus

This tutorial shows how the Amorpheus type-heterogeneous transformer handles
cross-domain transfer between building types.

**Runnable script:** `tutorials/run_cross_domain.py`

---

## Overview

Cross-domain transfer trains on one building type and tests on a **different**
type.  The Amorpheus architecture uses the morphology graph to apply
per-node-type encoders/decoders, enabling a single policy to work across
building types with different observation and action dimensions.

---

## What the script covers

1. **Inspect the benchmark** — `b2b.benchmarks.CrossDomainGeneralization(
   difficulty="easy", task="task_const_e0")` shows `train_type` and
   `test_type`; compare their observation and action dimensions.

2. **Explore the morphology** — `env.metadata["morphology"]` lists nodes with
   their `node_type`, `observation_dim`, and `action_dim`.

3. **Instantiate Amorpheus** — `AmorpheusPolicy(morphology=morph,
   building_type=..., d_model=64)` creates a policy that is type-aware.  A
   forward pass returns an `(action_distribution, values)` pair for use with
   PPO; the policy also exposes the SB3-style `predict(obs, deterministic)`
   interface for evaluation.

4. **Transfer to a different building type** — reassign `policy.morphology`
   to a new building's graph; the same learned parameters apply.

5. **Train with the CLI**:

    ```bash
    python -m baselines.train_cross_domain experiment=train_cross_domain
    ```

6. **Evaluate** — score the trained policy on held-out building types:

    ```bash
    python -m baselines.eval_cross_domain \
        --model-path outputs/cross_domain/amorpheus_policy.pt \
        --test-building-types Warehouse SingleFamilyHouse \
        --task task_occ_e0 --n-test 5
    ```

---

For the full programmatic walkthrough, run:

```bash
python tutorials/run_cross_domain.py
```
