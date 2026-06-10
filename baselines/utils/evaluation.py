"""Evaluation utilities for running rollouts and computing metrics."""

from __future__ import annotations

import logging
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


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


def run_episode_reward_only(
    env: Any,
    policy: PolicyLike,
    *,
    deterministic: bool = True,
) -> float:
    """Run one episode and return only the summed reward.

    Memory-light alternative to :func:`run_episode` for use in hyper-parameter
    tuning loops: no observations, actions, rewards, or infos are retained.
    A full-year 15-min simulation produces ~35k timesteps, so the info dict
    list alone dominates per-trial memory; dropping it is essential when
    running hundreds of trials in a single process.

    Args:
        env: A Gymnasium environment.
        policy: Object with ``predict(obs, deterministic) -> (action, _)``.
        deterministic: Whether to use deterministic actions.

    Returns:
        Sum of per-step rewards over the episode.
    """
    obs, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = bool(terminated) or bool(truncated)
    return total


def close_env_aggressively(
    env: Any,
    *,
    cleanup_dir: Path | None = None,
    thread_join_timeout: float = 10.0,
) -> None:
    """Compatibility shim — use ``env.close()`` directly.

    .. deprecated::
        Upstream ``minergym.environment.EnergyPlusEnvironment.close()`` is
        now leak-free: it stops the simulation thread, joins it, releases
        the native EnergyPlus state, and removes the output directory.
        Plain ``env.close()`` therefore handles everything that this helper
        used to do manually.

        This function will be removed in phase D.  Callers should switch to
        ``env.close()``.  The ``cleanup_dir`` argument is redundant when the
        environment was created via :func:`~building2building.api.make_env`
        or :func:`~building2building.envs.factory.make_env_from_config`
        (both track the output directory on the env); it is honoured here only
        as a fallback for envs constructed through other paths.

    Args:
        env: The environment to close, possibly wrapped.
        cleanup_dir: Directory to ``shutil.rmtree`` as a fallback when the
            env does not track its own output directory.  Redundant for envs
            created via :func:`~building2building.api.make_env`.
        thread_join_timeout: Ignored; the join timeout is now configured on
            the env via the upstream ``thread_join_timeout`` constructor
            parameter.  Kept for API compatibility.
    """
    warnings.warn(
        "close_env_aggressively() is deprecated; env.close() is now "
        "leak-free (thread join + native-state release + output-dir cleanup). "
        "This function will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    env.close()
    # Fallback: honour an explicit cleanup_dir in case the env was built
    # through a path that does not set _b2b_eplus_output_dir (e.g. a direct
    # create_simulator call predating this fix).  shutil.rmtree with
    # ignore_errors=True is safe if env.close() already removed the dir.
    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)


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
