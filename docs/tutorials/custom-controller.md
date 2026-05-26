# Write a Custom Controller

This tutorial shows how to write a custom reactive controller that works with
the B2B environment and evaluation infrastructure.

**Runnable script:** `tutorials/custom_controller.py`

---

## The Controller interface

All B2B controllers implement:

```python
class PolicyLike:
    def predict(self, obs: np.ndarray, deterministic: bool = True
                ) -> tuple[np.ndarray, Any]: ...
```

Optional extension points:

| Method | Purpose |
|---|---|
| `bind_env(env)` | Called after env creation; reads `env.metadata` (observation/action names, equipment list). |
| `reset()` | Called at the start of each episode. |
| `step_metrics(obs, *, action)` | Returns a `dict[str, float]` of per-step diagnostic values. |

---

## What the script covers

1. **Minimal constant controller** — `predict()` returns a fixed action
   vector; `bind_env()` reads `env.action_space.shape[0]`.

2. **Proportional controller** — `bind_env()` parses `env.metadata[
   "observation_names"]` and `"action_names"]` to find temperature and
   fan-speed indices; `predict()` applies a P-law.

3. **Run and score** — wrap the controller in `b2b.rollout()` or a manual
   loop; pass the `Trajectory` to `b2b.compute_normalized_score()`.

4. **Evaluate across buildings** — loop over `split="test"` indices with
   `b2b.new_make_env(...)`.

---

## Minimal example

```python
import numpy as np
import building2building as b2b


class ConstantController:
    def bind_env(self, env: b2b.Controller) -> None:
        self._n_act = env.action_space.shape[0]

    def predict(self, obs: np.ndarray, deterministic: bool = True
                ) -> tuple[np.ndarray, None]:
        return np.zeros(self._n_act, dtype=np.float32), None


env = b2b.new_make_env("SingleFamilyHouse", split="test", index=0, task="task_const_e0")
ctrl = ConstantController()
ctrl.bind_env(env)
traj = b2b.rollout(env, controller=b2b.callable_controller(ctrl.predict))
print(f"Return: {sum(traj.rewards):.1f}")
env.close()
```

For the full proportional-controller example, run:

```bash
python tutorials/custom_controller.py
```
