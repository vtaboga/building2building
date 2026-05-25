# Write a Custom Controller

This tutorial shows how to write a custom reactive controller that works with
the B2B environment and evaluation infrastructure.

A standalone script is available at `tutorials/custom_controller.py`.

---

## The Policy Interface

All B2B policies follow this interface (compatible with SB3):

```python
class PolicyLike:
    def predict(self, obs, deterministic=True) -> tuple[np.ndarray, Any]: ...
```

Optional methods:

| Method | Purpose |
|---|---|
| `bind_env(env)` | Called after env creation to read metadata |
| `reset()` | Called at the start of each episode |
| `step_metrics(obs, *, action)` | Return per-step metrics for logging |

## Step 1: Minimal Controller

```python
import numpy as np


class ConstantController:
    """Always requests 21 C supply air temperature with moderate fan speed."""

    def __init__(self, target_temp: float = 21.0, fan_fraction: float = 0.5):
        self.target_temp = target_temp
        self.fan_fraction = fan_fraction
        self._n_actions: int = 2

    def bind_env(self, env) -> None:
        self._n_actions = env.action_space.shape[0]

    def predict(self, obs, deterministic=True) -> tuple[np.ndarray, None]:
        action = np.zeros(self._n_actions, dtype=np.float32)
        action[0] = self.fan_fraction
        action[1] = self.target_temp
        return action, None
```

## Step 2: Run the Controller

```python
import building2building as b2b
from baselines.utils.evaluation import run_episode

env = b2b.new_make_env("SingleFamilyHouse", split="test", index=0, task="task_const_e0")
policy = ConstantController(target_temp=21.0, fan_fraction=0.5)
policy.bind_env(env)

result = run_episode(env, policy)
print(f"Episode return: {result.total_reward:.1f}")
print(f"Episode length: {result.episode_length}")
env.close()
```

## Step 3: Proportional Controller with Metadata

A smarter controller that reads observation names from metadata:

```python
import numpy as np
from typing import Any


class ProportionalController:
    """P-controller that adjusts fan speed based on temperature error."""

    def __init__(self, target_temp: float = 21.0, kp: float = 0.1):
        self.target_temp = target_temp
        self.kp = kp
        self._obs_names: list[str] = []
        self._act_names: list[str] = []
        self._temp_indices: list[int] = []
        self._fan_indices: list[int] = []
        self._sat_indices: list[int] = []

    def bind_env(self, env: Any) -> None:
        self._obs_names = env.metadata["observation_names"]
        self._act_names = env.metadata["action_names"]

        for i, name in enumerate(self._obs_names):
            if "zone air temperature" in name.lower():
                self._temp_indices.append(i)

        for i, name in enumerate(self._act_names):
            if "fan air mass flow rate" in name.lower():
                self._fan_indices.append(i)
            elif "schedule value" in name.lower():
                self._sat_indices.append(i)

    def predict(self, obs, deterministic=True) -> tuple[np.ndarray, None]:
        action = np.zeros(len(self._act_names), dtype=np.float32)

        for temp_idx, fan_idx in zip(self._temp_indices, self._fan_indices):
            error = self.target_temp - float(obs[temp_idx])
            fan_cmd = np.clip(0.3 + self.kp * abs(error), 0.0, 1.0)
            action[fan_idx] = fan_cmd

        for sat_idx in self._sat_indices:
            action[sat_idx] = self.target_temp

        return action, None

    def step_metrics(self, obs, *, action) -> dict[str, float]:
        temps = [float(obs[i]) for i in self._temp_indices]
        return {
            "target_temp": self.target_temp,
            "mean_zone_temp": np.mean(temps) if temps else 0.0,
        }
```

## Step 4: Compare Controllers

```python
env = b2b.new_make_env("OfficeSmall", split="test", index=0, task="task_const_e0")

for Controller, kwargs in [
    (ConstantController, {"target_temp": 21.0, "fan_fraction": 0.5}),
    (ProportionalController, {"target_temp": 21.0, "kp": 0.15}),
]:
    policy = Controller(**kwargs)
    policy.bind_env(env)
    result = run_episode(env, policy)
    print(f"{Controller.__name__}: return={result.total_reward:.1f}")

env.close()
```

## Step 5: Evaluate Across Buildings

```python
import numpy as np

policy = ProportionalController(target_temp=21.0, kp=0.15)
returns = []

for idx in range(5):
    env = b2b.new_make_env("OfficeSmall", split="test", index=idx, task="task_const_e0",
                           run_period="winter")
    policy.bind_env(env)
    result = run_episode(env, policy)
    returns.append(result.total_reward)
    env.close()

print(f"Mean return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
```
