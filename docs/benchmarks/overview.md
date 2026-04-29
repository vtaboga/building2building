# Benchmark Overview

## Motivation

Reinforcement learning for building control has shown promise, yet progress is
hampered by the lack of diverse, standardized evaluation protocols. Most
existing benchmarks offer only a handful of buildings and a single evaluation
axis, making it difficult to study **transfer**, **generalization**, and
**robustness**.

Building2Building fills this gap with four complementary benchmark problems,
each targeting a different dimension of policy generalization.

## Benchmark Problems

| Benchmark | Question | What Varies | What Stays Fixed |
|---|---|---|---|
| [Dynamics Adaptation](dynamics-adaptation.md) | Can a policy generalize across different building instances? | Building dynamics | Reward, action space |
| [Cross-Domain Generalization](cross-domain.md) | Does a policy transfer from type A to type B? | Building type | Reward, action space |
| [Goal Adaptation](goal-adaptation.md) | Can a policy adapt to a new reward/task? | Reward function | Building, action space |
| [Action-Space Transfer](action-transfer.md) | Can a policy adapt when actuators change? | Controllable actuators | Building, reward |

## Common API

All benchmarks inherit from `BenchmarkProblem` and expose:

```python
class BenchmarkProblem(ABC):
    def make_train_envs(self, n: int | None = None) -> list[gym.Env]: ...
    def make_test_envs(self, n: int | None = None) -> list[gym.Env]: ...
```

Usage:

```python
import building2building as b2b

bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task1")
train_envs = bench.make_train_envs(n=4)
test_envs = bench.make_test_envs(n=4)
```

## Evaluation Principles

1. **Reproducible splits** -- building selections are deterministic and
   seeded.
2. **Named task presets** -- `"task1"` through `"task5"` reproduce exact
   paper conditions.
3. **Normalized scoring** -- `b2b.compute_normalized_score()` compares agent
   returns to the reactive-controller baseline for the same building.
4. **Season-aware tasks** -- each side can specify a `run_period` (`winter`,
   `summer`, or `full_year`).
