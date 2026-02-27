"""Rollout runner and policy protocols for benchmark evaluation.

Provides :func:`run_rollout` and :func:`run_episode` which execute a
:class:`PolicyLike` on a Gymnasium environment and record results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


class PolicyLike(Protocol):
    """
    Minimal policy interface for benchmarking.

    We assume that the policy output dimension matches the
    environment action space shape.
    """

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]: ...


@runtime_checkable
class SupportsBindEnv(Protocol):
    def bind_env(self, env: Any) -> None: ...


@runtime_checkable
class SupportsReset(Protocol):
    def reset(self) -> None: ...


@runtime_checkable
class SupportsStepMetrics(Protocol):
    def step_metrics(self, obs: Any, *, action: np.ndarray) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    total_reward: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class RolloutData:
    """
    Flat rollout record for analysis/IO.

    Arrays are concatenated across episodes.
    """

    episode: np.ndarray  # (T,)
    step: np.ndarray  # (T,)
    reward: np.ndarray  # (T,)
    obs: np.ndarray  # (T, obs_dim)
    action: np.ndarray  # (T, act_dim)
    metrics: dict[str, np.ndarray]  # scalar metrics per step, (T,)


def _maybe_bind_env(*, policy: PolicyLike, env: Any) -> None:
    if isinstance(policy, SupportsBindEnv):
        # If a policy exposes bind_env(), failures should be surfaced: several
        # baseline policies require env metadata/action_names to operate.
        policy.bind_env(env)


def _maybe_reset_policy(*, policy: PolicyLike) -> None:
    if isinstance(policy, SupportsReset):
        try:
            policy.reset()
        except Exception:
            pass


def run_episode(
    *,
    env: Any,
    policy: PolicyLike,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> EpisodeResult:
    obs, _info = env.reset()
    done = False
    total_reward = 0.0
    n_steps = 0

    _maybe_reset_policy(policy=policy)

    cap = int(max_steps) if isinstance(max_steps, int) and max_steps > 0 else None

    while not done and (cap is None or n_steps < cap):
        action, _state = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
        total_reward += float(reward)
        done = bool(terminated or truncated)
        n_steps += 1

    return EpisodeResult(total_reward=float(total_reward), n_steps=int(n_steps))


def run_rollout(
    *,
    env: Any,
    policy: PolicyLike,
    n_episodes: int = 1,
    deterministic: bool = True,
    max_steps: int | None = None,
    record: bool = True,
) -> tuple[list[EpisodeResult], RolloutData | None]:
    """
    Run N episodes and optionally record a flat trajectory.

    The env-policy interaction stays strictly Gymnasium-standard:
      obs -> policy.predict(obs) -> env.step(action)
    Any policy-specific patches should live behind the `policy` adapter
    (e.g. bind_env/reset/step_metrics), not in the loop.
    """
    _maybe_bind_env(policy=policy, env=env)

    results: list[EpisodeResult] = []
    if not record:
        for _ in range(int(n_episodes)):
            results.append(
                run_episode(
                    env=env,
                    policy=policy,
                    deterministic=deterministic,
                    max_steps=max_steps,
                )
            )
        return results, None

    episodes: list[int] = []
    steps: list[int] = []
    rewards: list[float] = []
    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    metric_cols: dict[str, list[float]] = {}

    for ep in range(int(n_episodes)):
        obs, _info = env.reset()
        _maybe_reset_policy(policy=policy)
        done = False
        step = 0
        total_reward = 0.0

        cap = int(max_steps) if isinstance(max_steps, int) and max_steps > 0 else None

        while not done and (cap is None or step < cap):
            action, _state = policy.predict(obs, deterministic=deterministic)
            act_arr = np.asarray(action, dtype=float).reshape(-1)
            obs_arr = np.asarray(obs, dtype=float).reshape(-1)

            if isinstance(policy, SupportsStepMetrics):
                try:
                    ms = policy.step_metrics(obs, action=act_arr)
                    for k, v in ms.items():
                        metric_cols.setdefault(str(k), []).append(float(v))
                except Exception:
                    pass

            obs2, reward, terminated, truncated, _info = env.step(act_arr)

            episodes.append(int(ep))
            steps.append(int(step))
            rewards.append(float(reward))
            obs_rows.append(obs_arr)
            act_rows.append(act_arr)

            total_reward += float(reward)
            done = bool(terminated or truncated)
            step += 1
            obs = obs2

        results.append(EpisodeResult(total_reward=float(total_reward), n_steps=int(step)))

    obs_mat = np.stack(obs_rows, axis=0) if obs_rows else np.zeros((0, 0), dtype=float)
    act_mat = np.stack(act_rows, axis=0) if act_rows else np.zeros((0, 0), dtype=float)

    metrics_arr: dict[str, np.ndarray] = {}
    for k, vs in metric_cols.items():
        if len(vs) == len(rewards):
            metrics_arr[str(k)] = np.asarray(vs, dtype=float)

    data = RolloutData(
        episode=np.asarray(episodes, dtype=np.int32),
        step=np.asarray(steps, dtype=np.int32),
        reward=np.asarray(rewards, dtype=float),
        obs=obs_mat.astype(float),
        action=act_mat.astype(float),
        metrics=metrics_arr,
    )
    return results, data


def run_policy_on_env(
    *,
    env: Any,
    policy: PolicyLike,
    n_episodes: int = 1,
    deterministic: bool = True,
) -> list[EpisodeResult]:
    results, _data = run_rollout(
        env=env,
        policy=policy,
        n_episodes=n_episodes,
        deterministic=deterministic,
        record=False,
    )
    return results

