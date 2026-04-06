"""SubprocVecEnv wrapper that periodically restarts workers to reclaim
leaked C memory from the EnergyPlus runtime."""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper

logger = logging.getLogger(__name__)


class RecyclingSubprocVecEnv(VecEnvWrapper):
    """Periodically kill and respawn all ``SubprocVecEnv`` workers.

    The EnergyPlus C library leaks memory on every ``new_state`` /
    ``delete_state`` cycle (i.e. every episode reset).  This wrapper
    transparently tears down the inner ``SubprocVecEnv`` and creates a
    fresh one every *recycle_every* vectorised steps, capping the total
    leaked memory per process.

    At the recycling boundary the wrapper signals truncation
    (``done=True`` with ``terminal_observation`` in ``info``) so the PPO
    rollout buffer handles it identically to a ``TimeLimit`` truncation.

    Parameters
    ----------
    env_fns:
        The same factory callables that were used to build the inner
        ``SubprocVecEnv``.
    recycle_every:
        Number of *vectorised* ``step_wait`` calls between recycling
        events.  A value equal to the episode length (e.g. 35 040 for a
        full-year 15-min simulation) recycles roughly once per episode.
    start_method:
        Multiprocessing start method forwarded to ``SubprocVecEnv``.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        recycle_every: int,
        start_method: str | None = None,
    ) -> None:
        self._env_fns = list(env_fns)
        self._recycle_every = recycle_every
        self._start_method = start_method
        self._step_count = 0

        inner = SubprocVecEnv(self._env_fns, start_method=start_method)
        super().__init__(inner)

    def _recycle(self) -> np.ndarray:
        """Close every worker and spawn fresh ones, returning reset obs."""
        logger.info(
            "Recycling %d SubprocVecEnv workers after %d steps",
            self.num_envs,
            self._step_count,
        )
        self.venv.close()
        self.venv = SubprocVecEnv(self._env_fns, start_method=self._start_method)
        self._step_count = 0
        return self.venv.reset()

    # -- forwarded interface --------------------------------------------------

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rews, dones, infos = self.venv.step_wait()
        self._step_count += 1

        if self._step_count >= self._recycle_every:
            for i in range(self.num_envs):
                if not dones[i]:
                    infos[i]["TimeLimit.truncated"] = True
                    infos[i]["terminal_observation"] = obs[i]
            fresh_obs = self._recycle()
            dones = np.ones(self.num_envs, dtype=bool)
            return fresh_obs, rews, dones, infos

        return obs, rews, dones, infos

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return self.venv.reset()

    def close(self) -> None:
        self.venv.close()
