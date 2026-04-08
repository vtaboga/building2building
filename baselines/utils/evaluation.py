"""Evaluation utilities for running rollouts and computing metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class PolicyLike(Protocol):
    """Minimal interface for a policy usable in rollouts."""

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]: ...


@dataclass
class EpisodeResult:
    """Summary of a single episode rollout."""

    total_reward: float
    episode_length: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    infos: list[dict[str, Any]]


def run_episode(
    env: Any,
    policy: PolicyLike,
    *,
    deterministic: bool = True,
) -> EpisodeResult:
    """Run a single episode from reset to termination/truncation.

    Args:
        env: A Gymnasium environment.
        policy: Object with ``predict(obs, deterministic) -> (action, _)``.
        deterministic: Whether to use deterministic actions.

    Returns:
        An :class:`EpisodeResult` with episode data.
    """
    obs, info = env.reset()
    done = False

    all_obs: list[np.ndarray] = [obs]
    all_actions: list[np.ndarray] = []
    all_rewards: list[float] = []
    all_infos: list[dict[str, Any]] = [info]

    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        all_obs.append(obs)
        all_actions.append(np.asarray(action))
        all_rewards.append(float(reward))
        all_infos.append(info)

    return EpisodeResult(
        total_reward=sum(all_rewards),
        episode_length=len(all_rewards),
        observations=np.array(all_obs),
        actions=np.array(all_actions) if all_actions else np.array([]),
        rewards=np.array(all_rewards),
        infos=all_infos,
    )


def compute_temperature_satisfaction(
    observations: np.ndarray,
    zone_temp_indices: list[int],
    target_temp: float,
    deadband: float,
) -> float:
    """Compute percentage of timesteps where zone temps are in the deadband.

    Args:
        observations: ``(T+1, obs_dim)`` array of observations.
        zone_temp_indices: Indices of zone temperature observations.
        target_temp: Target temperature (Celsius).
        deadband: Half-width of the acceptable band (Celsius).

    Returns:
        Fraction in [0, 1].
    """
    if observations.ndim != 2 or not zone_temp_indices:
        return 0.0

    zone_temps = observations[:, zone_temp_indices]
    within = np.abs(zone_temps - target_temp) <= deadband
    return float(within.mean())
