from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from b2b.benchmark.adaptive_dynamics import (
    AdaptiveDynamicsRecord,
    benchmark_adaptive_dynamics,
)
from b2b.benchmark.runner import PolicyLike


def _default_base_config() -> dict[str, Any]:
    """
    Minimal config that works with `b2b.make_env.make_env()`.

    The adaptive dynamics benchmark overrides `bldg.selection` internally to select
    specific Hydro-Québec buildings.
    """
    return {
        "env": {
            "control_mode": "hvac_actuators",
            # The benchmark runs its own (potentially long) episode loop; env.max_steps
            # is typically ignored by the simulator wrappers, but keep it present for
            # compatibility with existing configs.
            "max_steps": 0,
            "normalize_obs": False,
        },
        "reward": {"reward_type": None},
        "bldg": {"bldg": {}},
    }


@dataclass(frozen=True, slots=True)
class AdaptiveDynamicsProblem:
    """
    Adaptive dynamics control problem on Hydro-Québec buildings.

    Intended usage from external repos:

    ```python
    from b2b.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem

    problem = AdaptiveDynamicsProblem()
    records = problem.run(policy)
    ```
    """

    split: Literal["train", "test", "test_small"] = "train"
    start: int = 0
    limit: int = 0
    max_steps: int | None = None
    base_config: dict[str, Any] | None = None

    def run(
        self,
        policy: PolicyLike,
        *,
        output_dir: str | Path = ".",
        env_wrapper: Callable[[gym.Env], gym.Env] | None = None,
    ) -> list[AdaptiveDynamicsRecord]:
        cfg: dict[str, Any] = (
            dict(self.base_config)
            if isinstance(self.base_config, dict)
            else _default_base_config()
        )

        bench_section: dict[str, Any] = {
            "split": str(self.split),
            "start": int(self.start),
            "limit": int(self.limit),
        }
        if self.max_steps is not None:
            bench_section["max_steps"] = int(self.max_steps)

        # This matches the existing config layout used by the benchmark function.
        cfg["benchmark"] = bench_section

        return benchmark_adaptive_dynamics(
            cfg, policy, output_dir=Path(output_dir), env_wrapper=env_wrapper
        )


def run(
    policy: PolicyLike,
    *,
    output_dir: str | Path = ".",
    split: Literal["train", "test", "test_small"] = "train",
    start: int = 0,
    limit: int = 0,
    max_steps: int | None = None,
    base_config: dict[str, Any] | None = None,
    env_wrapper: Callable[[gym.Env], gym.Env] | None = None,
) -> list[AdaptiveDynamicsRecord]:
    """
    Convenience wrapper so external code can call `problem_adaptive_dynamics.run(policy)`.
    """
    return AdaptiveDynamicsProblem(
        split=split,
        start=start,
        limit=limit,
        max_steps=max_steps,
        base_config=base_config,
    ).run(policy, output_dir=output_dir, env_wrapper=env_wrapper)
