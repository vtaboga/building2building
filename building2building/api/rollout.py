"""Trajectory capture for Building2Building environments.

This module provides a pure-capture rollout utility: it runs one episode of a
controller in an environment and returns every tensor needed to analyse the
run offline. It deliberately does **not** compute metrics, plot anything, or
decompose the reward. Those concerns belong to user-level analysis code.

The captured :class:`Trajectory` preserves both the flat gym-visible
observation (``observations``) and the nested EnergyPlus-native observation
(``raw_observations``). The latter is what the built-in reward functions in
:mod:`building2building.simulator.rewards` read (e.g.
``obs["temperature"][zone]``, ``obs["energy"]["electricity"]``), so keeping it
intact lets you recompute reward components — temperature error, energy
penalty, etc. — offline without any lossy re-derivation.

Example::

    import building2building as b2b
    import numpy as np

    env = b2b.make_env("OfficeSmall", task="task_const_e0", run_period="winter")

    def random_controller(_obs):
        return env.action_space.sample()

    traj = b2b.rollout(env, random_controller, seed=0)
    traj.to_npz("rollout.npz")
    reloaded = b2b.Trajectory.from_npz("rollout.npz")
"""

from __future__ import annotations

import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np

from building2building.data.registry import BuildingInfo
from building2building.types import TaskConfig

__all__ = [
    "Controller",
    "Trajectory",
    "callable_controller",
    "rollout",
]


@runtime_checkable
class Controller(Protocol):
    """Protocol for closed-loop controllers usable by :func:`rollout`.

    A minimal controller is a plain callable ``fn(obs) -> action``; use
    :func:`callable_controller` to adapt one into this protocol. Stateful
    controllers should implement :meth:`reset` to clear any per-episode state
    at the start of a new rollout.
    """

    def reset(self, env: gym.Env) -> None:  # pragma: no cover - optional
        """Reset per-episode state. Default implementation is a no-op."""

    def __call__(self, observation: Any) -> Any:
        """Return the next action given the current observation."""


class _CallableController:
    """Wrap a plain ``fn(obs) -> action`` into the :class:`Controller` protocol."""

    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self._fn = fn

    def reset(self, env: gym.Env) -> None:
        del env

    def __call__(self, observation: Any) -> Any:
        return self._fn(observation)


def callable_controller(fn: Callable[[Any], Any]) -> Controller:
    """Wrap a plain callable into the :class:`Controller` protocol."""
    return _CallableController(fn)


def _as_controller(
    controller: Controller | Callable[[Any], Any],
) -> Controller:
    if isinstance(controller, Controller):
        return controller
    return callable_controller(controller)


@dataclass
class Trajectory:
    """Per-step arrays plus episode-level context captured by :func:`rollout`.

    Attributes:
        observations: Flat gym-visible observations. Shape ``(T + 1, obs_dim)``
            for box-shaped observation spaces, or an object array of length
            ``T + 1`` for structured spaces. The first row is the reset
            observation; subsequent rows are post-step observations.
        actions: Actions taken by the controller. Shape ``(T, action_dim)``
            for box-shaped action spaces, or an object array of length ``T``.
        rewards: Per-step scalar rewards. Shape ``(T,)``.
        terminateds: Whether each step triggered termination. Shape ``(T,)``.
        truncateds: Whether each step triggered truncation. Shape ``(T,)``.
        infos: Per-step ``info`` dicts with ``raw_observation`` pulled out
            (see ``raw_observations``).
        raw_observations: Nested EnergyPlus-native observations, one per step
            including the reset. These carry ``temperature[zone]``,
            ``energy[...]``, and any ``target_temperature`` fields and are
            what the built-in reward functions read.
        controlled_zones: Zones that the reward function considers
            (``env.metadata["controlled_zones"]``).
        task_config: Task configuration passed to the environment
            (``env.metadata["task_config"]``).
        building_info: Resolved building metadata (includes ``climate_zone``,
            ``weather_file``, ``num_zones``, ...). ``None`` if the env was
            built without the standard factory.
        observation_names: Per-slot names for the flat observation (copied
            from ``env.metadata["observation_names"]`` when available).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminateds: np.ndarray
    truncateds: np.ndarray
    infos: list[dict[str, Any]]
    raw_observations: list[Any]
    controlled_zones: list[str]
    task_config: TaskConfig | None
    building_info: BuildingInfo | None
    observation_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.rewards.shape[0])

    def to_npz(self, path: str | Path) -> None:
        """Serialize the trajectory to a ``.npz`` file.

        Nested / dataclass fields (``raw_observations``, ``infos``,
        ``task_config``, ``building_info``) are pickled through a single
        ``object`` array so the file can be reloaded with
        :meth:`Trajectory.from_npz` without external schema.
        """
        path = Path(path)
        extras = pickle.dumps(
            {
                "raw_observations": self.raw_observations,
                "infos": self.infos,
                "task_config": self.task_config,
                "building_info": self.building_info,
                "controlled_zones": list(self.controlled_zones),
                "observation_names": list(self.observation_names),
            }
        )
        np.savez(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminateds=self.terminateds,
            truncateds=self.truncateds,
            extras=np.frombuffer(extras, dtype=np.uint8),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "Trajectory":
        """Reload a trajectory previously saved with :meth:`to_npz`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            observations = np.asarray(data["observations"])
            actions = np.asarray(data["actions"])
            rewards = np.asarray(data["rewards"])
            terminateds = np.asarray(data["terminateds"])
            truncateds = np.asarray(data["truncateds"])
            extras_bytes = bytes(data["extras"].tobytes())
        extras = pickle.loads(extras_bytes)
        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            infos=extras["infos"],
            raw_observations=extras["raw_observations"],
            controlled_zones=extras["controlled_zones"],
            task_config=extras["task_config"],
            building_info=extras["building_info"],
            observation_names=extras.get("observation_names", []),
        )


def _episode_length_upper_bound(env: gym.Env, max_steps: int | None) -> int:
    if max_steps is not None:
        return int(max_steps)
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None):
        return int(spec.max_episode_steps)
    # Fallback: look at TimeLimit wrapper attribute.
    max_limit = getattr(env, "_max_episode_steps", None)
    if max_limit is not None:
        return int(max_limit)
    task_config = env.unwrapped.metadata.get("task_config")
    if task_config is not None:
        return int(task_config.expected_steps())
    raise ValueError(
        "Cannot determine rollout length: pass `max_steps` explicitly, or "
        "wrap the env in gymnasium.wrappers.TimeLimit."
    )


def rollout(
    env: gym.Env,
    controller: Controller | Callable[[Any], Any],
    *,
    max_steps: int | None = None,
    seed: int | None = None,
    stop_on_terminated: bool = True,
) -> Trajectory:
    """Run one episode and return a :class:`Trajectory`.

    The rollout is a pure capture: this function does **not** close the env
    (caller controls the lifecycle), does **not** compute any summary metric,
    and does **not** decompose the reward. It only records what the env and
    the controller produced.

    Args:
        env: Gymnasium environment (typically created via
            :func:`building2building.make_env`).
        controller: Either a :class:`Controller` instance or a plain callable
            ``fn(obs) -> action``. If it exposes a ``reset(env)`` method, it
            is called once after ``env.reset``.
        max_steps: Episode length cap. If ``None``, inferred from
            ``env.spec.max_episode_steps`` or the TimeLimit wrapper.
        seed: Seed passed to ``env.reset``.
        stop_on_terminated: If ``True`` (default), break on the first
            ``terminated`` or ``truncated`` step. If ``False``, keep stepping
            until ``max_steps``.

    Returns:
        The full :class:`Trajectory`.
    """
    controller = _as_controller(controller)

    obs, info = env.reset(seed=seed)
    controller.reset(env)

    steps_cap = _episode_length_upper_bound(env, max_steps)

    meta = env.unwrapped.metadata
    controlled_zones = list(meta.get("controlled_zones", []))
    task_config = meta.get("task_config")
    building_info = meta.get("building_info")
    observation_names = list(meta.get("observation_names", []))

    observations_list: list[Any] = [deepcopy(obs)]
    raw_observations: list[Any] = [deepcopy(info.get("raw_observation"))]
    actions_list: list[Any] = []
    rewards_list: list[float] = []
    terminateds_list: list[bool] = []
    truncateds_list: list[bool] = []
    # Keep a copy of info minus `raw_observation` (stored separately).
    infos_list: list[dict[str, Any]] = [
        {k: v for k, v in info.items() if k != "raw_observation"}
    ]

    for _ in range(steps_cap):
        action = controller(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        observations_list.append(deepcopy(obs))
        raw_observations.append(deepcopy(info.get("raw_observation")))
        infos_list.append({k: v for k, v in info.items() if k != "raw_observation"})
        actions_list.append(np.asarray(action))
        rewards_list.append(float(reward))
        terminateds_list.append(bool(terminated))
        truncateds_list.append(bool(truncated))
        if stop_on_terminated and (terminated or truncated):
            break

    try:
        observations = np.asarray(observations_list)
    except (ValueError, TypeError):
        observations = np.array(observations_list, dtype=object)
    try:
        actions = np.asarray(actions_list)
    except (ValueError, TypeError):
        actions = np.array(actions_list, dtype=object)

    return Trajectory(
        observations=observations,
        actions=actions,
        rewards=np.asarray(rewards_list, dtype=np.float64),
        terminateds=np.asarray(terminateds_list, dtype=bool),
        truncateds=np.asarray(truncateds_list, dtype=bool),
        infos=infos_list,
        raw_observations=raw_observations,
        controlled_zones=controlled_zones,
        task_config=task_config,
        building_info=building_info,
        observation_names=observation_names,
    )
