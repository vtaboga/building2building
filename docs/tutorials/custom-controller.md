# Write a Custom Controller

This tutorial shows how to write a custom reactive controller that works with
the B2B environment and evaluation infrastructure.

**Runnable script:** `tutorials/custom_controller.py`

---

## The Controller interface

The baselines' evaluation helpers (`baselines.utils.evaluation.run_episode`)
accept anything with the SB3-style `predict` interface:

```python
class PolicyLike:
    def predict(self, obs: np.ndarray, deterministic: bool = True
                ) -> tuple[np.ndarray, Any]: ...
```

`b2b.rollout()` instead expects a plain callable `fn(obs) -> action`
(optionally with a `reset(env)` method) — adapt a `predict`-style controller
with `lambda obs: ctrl.predict(obs)[0]`.

Conventions used by the baselines' reference controllers (you call these
yourself; nothing invokes them automatically):

| Method | Purpose |
|---|---|
| `bind_env(env)` | Call after env creation; reads `env.metadata` (observation/action names, equipment list) to wire up indices. |
| `reset()` | Call at the start of each episode to clear per-episode state (e.g. integrator terms). |

---

## What the script covers

1. **Constant controller** — `predict()` returns a fixed action vector;
   `bind_env()` parses `env.metadata["action_names"]` so the fixed fan
   command and setpoint temperature land on the right actuators for any
   building.

2. **Proportional controller** — `bind_env()` parses `env.metadata[
   "observation_names"]` and `"action_names"]` to find temperature and
   fan-speed indices; `predict()` applies a P-law.

3. **Run and score** — run the controller with `b2b.rollout()` or a manual
   loop; sum the per-step rewards and pass the cumulative return to
   `b2b.compute_normalized_score(...)`.  Both agent and baseline returns are
   negative, so lower is better: a score below 1.0 beats the reactive
   baseline.

4. **Evaluate across buildings** — loop over `split="test"` indices with
   `b2b.make_env(...)`.

---

## Minimal example

```python
import gymnasium as gym
import numpy as np
import building2building as b2b


class ConstantController:
    def bind_env(self, env: gym.Env) -> None:
        self._n_act = env.action_space.shape[0]

    def predict(self, obs: np.ndarray, deterministic: bool = True
                ) -> tuple[np.ndarray, None]:
        return np.zeros(self._n_act, dtype=np.float32), None


env = b2b.make_env("SingleFamilyHouse", split="test", index=0, task="task_const_e0")
ctrl = ConstantController()
ctrl.bind_env(env)
traj = b2b.rollout(env, lambda obs: ctrl.predict(obs)[0])
print(f"Return: {traj.rewards.sum():.1f}")
env.close()
```

For the full proportional-controller example, run:

```bash
python tutorials/custom_controller.py
```
