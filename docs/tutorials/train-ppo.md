# Train a PPO Agent

This tutorial trains a PPO agent on a single building using Stable-Baselines3,
evaluates it, and computes a normalized score.

A standalone script is available at `tutorials/train_ppo_specialist.py`.

---

## Prerequisites

```bash
pip install -e ".[training]"
```

## Step 1: Create the Environment

```python
import building2building as b2b
from stable_baselines3 import PPO

env = b2b.new_make_env(
    "OfficeSmall",
    split="train",
    index=0,
    task="task1",
    run_period="winter",  # shorter episodes for faster training
)
env = b2b.NormalizeObservation(env)

print(f"Obs dim: {env.observation_space.shape[0]}")
print(f"Act dim: {env.action_space.shape[0]}")
```

## Step 2: Train PPO

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    learning_rate=3e-4,
)
model.learn(total_timesteps=50_000)
model.save("ppo_office_small_winter")
env.close()
```

## Step 3: Evaluate

```python
eval_env = b2b.new_make_env(
    "OfficeSmall",
    split="test",
    index=0,
    task="task1",
    run_period="winter",
)
eval_env = b2b.NormalizeObservation(eval_env)

model = PPO.load("ppo_office_small_winter")

obs, _ = eval_env.reset()
total_reward = 0.0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Test episode return: {total_reward:.2f}")
eval_env.close()
```

## Step 4: Compare with Baseline

Run the rule-based controller on the same building to generate a reference:

```bash
python -m baselines.run_rule_based experiment=eval_rule_based \
    building_types=[OfficeSmall] tasks=[task1] max_buildings_per_type=1
```

Then compute the normalized score:

```python
score = b2b.compute_normalized_score(
    cumulative_return=total_reward,
    building_type="OfficeSmall",
    task="task1",
)
print(f"Normalized score: {score:.3f}")
```

A score above 1.0 means the PPO agent outperforms the reactive baseline.

## Using the Hydra CLI

The same experiment can be run via the baselines CLI:

```bash
python -m baselines.train_ppo experiment=train_ppo \
    building_types=[OfficeSmall] tasks=[task1] \
    buildings_per_type=1 training.total_timesteps=50000
```

## Tips

- Use `run_period="winter"` for faster iteration during development
- Increase `total_timesteps` to 500k--1M for meaningful results
- Monitor with W&B: add `wandb.enabled=true` to the CLI
