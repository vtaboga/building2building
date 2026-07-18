#!/usr/bin/env python3
"""Write and evaluate a custom controller.

Implements a simple proportional controller and evaluates it on multiple
buildings, demonstrating the policy interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import building2building as b2b
from baselines.utils.evaluation import run_episode


class ConstantController:
    """Always requests a fixed setpoint temperature and fan command."""

    def __init__(self, target_temp: float = 21.0, fan_fraction: float = 0.5):
        self.target_temp = target_temp
        self.fan_fraction = fan_fraction
        self._n_actions: int = 0
        self._fan_indices: list[int] = []
        self._sat_indices: list[int] = []

    def bind_env(self, env: Any) -> None:
        act_names = env.metadata["action_names"]
        self._n_actions = len(act_names)
        self._fan_indices = [
            i for i, n in enumerate(act_names) if "fan air mass flow rate" in n.lower()
        ]
        self._sat_indices = [
            i for i, n in enumerate(act_names) if "schedule value" in n.lower()
        ]

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        action = np.zeros(self._n_actions, dtype=np.float32)
        for i in self._fan_indices:
            action[i] = self.fan_fraction
        for i in self._sat_indices:
            action[i] = self.target_temp
        return action, None


class ProportionalController:
    """P-controller: adjusts fan speed based on temperature error."""

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

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        action = np.zeros(len(self._act_names), dtype=np.float32)

        for temp_idx, fan_idx in zip(self._temp_indices, self._fan_indices):
            error = self.target_temp - float(obs[temp_idx])
            fan_cmd = float(np.clip(0.3 + self.kp * abs(error), 0.0, 1.0))
            action[fan_idx] = fan_cmd

        for sat_idx in self._sat_indices:
            action[sat_idx] = self.target_temp

        return action, None


def main() -> None:
    building_type = "OfficeSmall"
    task = "task_const_e0"
    run_period = "winter"

    # -- Compare controllers on a single building --
    print(f"Comparing controllers on {building_type}/{task} ({run_period})")
    print("-" * 50)

    env = b2b.make_env(
        building_type, split="test", index=0, task=task, run_period=run_period
    )

    controllers = [
        ("Constant(21C, 50%)", ConstantController(target_temp=21.0, fan_fraction=0.5)),
        ("Proportional(21C, kp=0.1)", ProportionalController(target_temp=21.0, kp=0.1)),
        ("Proportional(21C, kp=0.2)", ProportionalController(target_temp=21.0, kp=0.2)),
    ]

    for name, policy in controllers:
        policy.bind_env(env)
        result = run_episode(env, policy)
        print(f"  {name}: return={result.total_reward:.1f}")

    env.close()

    # -- Evaluate best controller across buildings --
    print(f"\nEvaluating ProportionalController across 5 test buildings...")
    returns: list[float] = []

    for idx in range(5):
        env = b2b.make_env(
            building_type, split="test", index=idx, task=task, run_period=run_period
        )
        policy = ProportionalController(target_temp=21.0, kp=0.15)
        policy.bind_env(env)
        result = run_episode(env, policy)
        returns.append(result.total_reward)
        env.close()

    print(f"  Mean return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")


if __name__ == "__main__":
    main()
