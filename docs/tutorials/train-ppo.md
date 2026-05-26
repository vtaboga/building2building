# Train a PPO Agent

This tutorial trains a PPO specialist on a single building using
Stable-Baselines3, evaluates it on a held-out test building, and computes a
normalized score.

**Runnable script:** `tutorials/train_ppo_specialist.py`

---

## Prerequisites

```bash
pip install -e ".[training]"
```

---

## What the script covers

1. **Pick a building and task** — `"OfficeSmall"`, `task="task_const_e0"`,
   `run_period="winter"` (shorter episodes for faster iteration).

2. **Create a training environment** — `b2b.new_make_env(...)` wrapped with
   `b2b.NormalizeObservation(env)`.

3. **Train PPO** — standard SB3 `PPO("MlpPolicy", env, ...)` loop; save with
   `model.save(...)`.

4. **Evaluate on a test building** — load the saved model, run one episode,
   collect the cumulative return.

5. **Compute a normalized score** — `b2b.compute_normalized_score(...)` returns
   a score relative to the reactive-controller baseline.  A score above 1.0
   means the PPO agent outperforms the reactive controller.

---

## Run via the Hydra CLI

The same experiment runs through the baselines CLI without writing Python:

```bash
python -m baselines.train_ppo experiment=train_ppo \
    building_types=[OfficeSmall] tasks=[task_const_e0] \
    buildings_per_type=1 training.total_timesteps=50000
```

For the full programmatic pipeline, run:

```bash
python tutorials/train_ppo_specialist.py
```
