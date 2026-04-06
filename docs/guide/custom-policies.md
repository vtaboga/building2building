# Using Custom Policies

B2B supports three ways to supply a control policy: hand-written baseline controllers, SB3 checkpoints, and arbitrary Python classes. This page describes the policy interface, how to write your own controller, and how to load policies via configuration.

---

## The Policy Interface

All B2B policies follow the same interface, modeled after Stable-Baselines3:

```python
class PolicyLike:
    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        """Select an action given an observation.

        Args:
            obs: Current observation from the environment.
            deterministic: If True, return the greedy action.

        Returns:
            A tuple of (action, state). The state is typically None
            for non-recurrent policies.
        """
        ...
```

### Optional Methods

Beyond `predict`, controllers can implement additional lifecycle methods:

| Method | Signature | Purpose |
|---|---|---|
| `bind_env(env)` | `env: gym.Env -> None` | Called once after environment creation. Use to read metadata, cache actuator indices, etc. |
| `reset()` | `-> None` | Called at the start of each episode |
| `step_metrics(obs, *, action)` | `-> dict[str, float]` | Return per-step metrics for logging (e.g., target temp, PID error) |

---

## Writing a Custom Controller

### Minimal Example

```python
import numpy as np


class ConstantTemperaturePolicy:
    """Always request 21°C supply air temperature with moderate fan speed."""

    def __init__(self, target_temp: float = 21.0, fan_fraction: float = 0.5):
        self.target_temp = target_temp
        self.fan_fraction = fan_fraction
        self._n_actions: int = 2

    def bind_env(self, env) -> None:
        self._n_actions = env.action_space.shape[0]

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        action = np.zeros(self._n_actions, dtype=np.float32)
        # Assuming unitary system: [fan_mass_flow, supply_air_temp]
        action[0] = self.fan_fraction
        action[1] = self.target_temp
        return action, None
```

### Full-Featured Controller

Here is a more complete example with environment binding, metrics, and observation reading:

```python
from dataclasses import dataclass
from typing import Any

import numpy as np


class ProportionalController:
    """PI-style controller that modulates fan speed based on temperature error."""

    def __init__(self, target_temp: float = 21.0, kp: float = 0.1):
        self.target_temp = target_temp
        self.kp = kp
        self._obs_names: list[str] = []
        self._act_names: list[str] = []
        self._temp_idx: int | None = None
        self._fan_idx: int | None = None
        self._sat_idx: int | None = None

    def bind_env(self, env: Any) -> None:
        """Read observation and action names from env metadata."""
        meta = env.metadata
        self._obs_names = meta["observation_names"]
        self._act_names = meta["action_names"]

        # Find zone temperature index
        for i, name in enumerate(self._obs_names):
            if "zone air temperature" in name.lower():
                self._temp_idx = i
                break

        # Find actuator indices
        for i, name in enumerate(self._act_names):
            if "fan air mass flow rate" in name.lower():
                self._fan_idx = i
            elif "schedule value" in name.lower():
                self._sat_idx = i

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        n_act = len(self._act_names)
        action = np.zeros(n_act, dtype=np.float32)

        if self._temp_idx is not None:
            current_temp = float(obs[self._temp_idx])
            error = self.target_temp - current_temp

            # Modulate fan speed based on error
            if self._fan_idx is not None:
                fan_cmd = np.clip(0.3 + self.kp * abs(error), 0.0, 1.0)
                action[self._fan_idx] = fan_cmd

            # Set supply air temperature
            if self._sat_idx is not None:
                if error > 0:
                    action[self._sat_idx] = min(self.target_temp + 5.0, 40.0)
                else:
                    action[self._sat_idx] = max(self.target_temp - 5.0, 12.0)

        return action, None

    def step_metrics(self, obs: Any, *, action: np.ndarray) -> dict[str, float]:
        metrics = {"target_temp_c": self.target_temp}
        if self._temp_idx is not None:
            obs_arr = np.asarray(obs, dtype=float).reshape(-1)
            metrics["zone_temp_c"] = float(obs_arr[self._temp_idx])
            metrics["temp_error"] = self.target_temp - float(obs_arr[self._temp_idx])
        return metrics
```

---

## Loading Policies via Configuration

### SB3 Checkpoint (`policy=sb3`)

Load a trained Stable-Baselines3 model from a `.zip` checkpoint:

```yaml
# configs/policy/sb3.yaml
type: sb3
algorithm: ppo              # SB3 / sb3-contrib algorithm name
checkpoint_path: ???        # Path to the saved .zip model file
```

```bash
python -m building2building.benchmark.baseline_rollout \
    policy=sb3 \
    policy.algorithm=ppo \
    policy.checkpoint_path=checkpoints/ppo_office_small.zip
```

Supported algorithms include any SB3 or sb3-contrib algorithm: `ppo`, `sac`, `dqn`, `trpo`, etc.

### Custom Python Class (`policy=custom`)

Load an arbitrary Python class that implements the `predict()` interface:

```yaml
# configs/policy/custom.yaml
type: custom
module: ???                 # Fully-qualified Python module path
class_name: ???             # Class name within the module
kwargs: {}                  # Keyword arguments passed to __init__
```

```bash
python -m building2building.benchmark.baseline_rollout \
    policy=custom \
    policy.module=my_package.controllers \
    policy.class_name=ProportionalController \
    policy.kwargs.target_temp=22.0 \
    policy.kwargs.kp=0.15
```

The custom class must be importable from the Python path. B2B will:

1. Import `policy.module`
2. Get the class `policy.class_name` from the module
3. Instantiate it with `policy.kwargs`
4. Call `bind_env(env)` if the method exists
5. Use `predict(obs, deterministic)` during rollout

---

## Built-In Baseline Controllers

B2B includes rule-based controllers for benchmarking:

### UnitaryG36Policy

G36-inspired PI airflow + Trim-and-Respond SAT controller for PSZ systems:

```bash
python scripts/baselines.py policy=unitary_g36
```

See [G36 Controller docs](../baselines/unitary-g36.md) for full parameter reference.

### AshraeAirLoopPolicy / AirLoopSatPolicy

ASHRAE-based controllers for VAV air-loop systems:

```bash
python scripts/baselines.py policy=ashrae_air_loop
python scripts/baselines.py policy=air_loop_sat
```

---

## Running a Rollout

The baseline rollout infrastructure runs any policy through a full episode and records observations, actions, rewards, and metrics:

```bash
python scripts/baselines.py \
    policy=unitary_g36 \
    bldg=single_family \
    n_episodes=1 \
    wandb.enabled=true
```

The rollout produces:

| Output | Format | Description |
|---|---|---|
| `rollout.csv` | CSV | Full trajectory with obs, actions, rewards, metrics |
| `rollout.npz` | NumPy | Compressed arrays for efficient post-processing |
| `config.json` | JSON | Resolved configuration snapshot |

When W&B is enabled, the rollout also logs:

- Temperature timeseries plots
- Action timeseries plots
- Energy consumption plots
- Reward curves
- Summary statistics (mean reward, episode return)

---

## Policy Comparison Workflow

A common workflow for comparing policies:

```bash
# 1. Run baseline
python scripts/baselines.py \
    policy=unitary_g36 \
    bldg=single_family \
    wandb.tags="[baseline]"

# 2. Run trained PPO
python scripts/baselines.py \
    policy=sb3 \
    policy.algorithm=ppo \
    policy.checkpoint_path=checkpoints/ppo.zip \
    bldg=single_family \
    wandb.tags="[ppo,trained]"

# 3. Run custom controller
python scripts/baselines.py \
    policy=custom \
    policy.module=my_controllers \
    policy.class_name=MySmartController \
    bldg=single_family \
    wandb.tags="[custom]"
```

All runs log to the same W&B project for easy comparison.

---

## Next Steps

- Learn about the full [configuration system](configuration.md) for Hydra overrides
- Explore [reward functions](rewards.md) to evaluate your policy
- Set up multi-building evaluation in [Getting Started](../getting-started.md)
