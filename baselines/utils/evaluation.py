"""Evaluation utilities for running rollouts and computing metrics."""

from __future__ import annotations

import gc
import logging
import shutil
import threading
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
    """Force-release EnergyPlus resources held by *env*.

    ``EnergyPlusEnvironment.close`` is a no-op inherited from
    ``gymnasium.Env``: calling it does **not** stop the simulation thread,
    does not free the allocated EnergyPlus state, and does not clean up the
    output directory.  When many envs are created sequentially inside a
    long-running process (e.g. Optuna tuning), the following all leak:

    * the daemon ``threading.Thread`` running ``api.runtime.run_energyplus``,
    * cyclic references between the ``EnergyPlusSimulation`` object, its
      ``StateStarted``/``StateDone`` state, and the thread's closure,
    * the EnergyPlus output directory, which under SLURM is usually
      ``$SLURM_TMPDIR`` (tmpfs, RAM-backed) and fills with ``eplusout.*``
      artefacts.

    This helper:

    1. Drills through gymnasium wrappers (e.g. ``TimeLimit``) to the
       underlying ``EnergyPlusEnvironment``.
    2. Calls ``try_stop`` on the simulation to signal shutdown.
    3. Joins the EnergyPlus thread so its closure (and the captured
       ``ManagedState``) become collectable.
    4. Drops the ``ep`` reference and triggers ``gc.collect()`` so that
       ``ManagedState.__del__`` fires and the native state is released.
    5. Removes *cleanup_dir* (the EnergyPlus output directory).

    Args:
        env: The environment returned by :func:`new_make_env`, possibly
            wrapped.
        cleanup_dir: Directory to ``shutil.rmtree`` after the thread has
            exited.  Should be the ``eplus_output_dir`` passed to the env.
        thread_join_timeout: Seconds to wait for the EnergyPlus thread
            before giving up and logging a warning.
    """
    try:
        env.close()
    except Exception as e:
        logger.debug("env.close() raised: %s", e)

    inner = env
    for _ in range(8):
        next_inner = getattr(inner, "env", None)
        if next_inner is None or next_inner is inner:
            break
        inner = next_inner

    sim = getattr(inner, "ep", None)
    if sim is not None:
        try:
            sim.try_stop()
        except Exception as e:
            logger.debug("sim.try_stop() raised: %s", e)

        state = getattr(sim, "state", None)
        ep_thread = getattr(state, "ep_thread", None)
        if isinstance(ep_thread, threading.Thread) and ep_thread.is_alive():
            ep_thread.join(timeout=thread_join_timeout)
            if ep_thread.is_alive():
                logger.warning(
                    "EnergyPlus thread did not exit within %.1fs; "
                    "resources may leak.",
                    thread_join_timeout,
                )

        try:
            inner.ep = None
        except Exception:
            pass

    gc.collect()

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
