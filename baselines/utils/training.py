"""SB3 training utilities for the baselines."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Sequence

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

logger = logging.getLogger(__name__)

PAPER_PPO_DEFAULTS: dict[str, Any] = {
    "learning_rate": 5e-5,
    "batch_size": 336,
    "n_steps": 672,
    "n_epochs": 5,
    "target_kl": 0.02,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.Tanh,
        "ortho_init": True,
        "log_std_init": -1.0,
    },
}


def build_ppo(
    env: VecEnv,
    *,
    tensorboard_log: str | None = None,
    seed: int | None = None,
    verbose: int = 1,
    **overrides: Any,
) -> PPO:
    """Create a PPO model with the paper's default hyperparameters.

    Any key from :data:`PAPER_PPO_DEFAULTS` can be overridden via
    keyword arguments.

    Args:
        env: Vectorized environment.
        tensorboard_log: TensorBoard log directory.
        seed: Random seed.
        verbose: SB3 verbosity level (0=silent, 1=info, 2=debug).
        **overrides: PPO constructor keyword overrides.

    Returns:
        A configured :class:`PPO` instance.
    """
    kwargs = {**PAPER_PPO_DEFAULTS, **overrides}

    policy_kwargs = dict(kwargs.pop("policy_kwargs", {}))
    if "activation_fn" in policy_kwargs and isinstance(
        policy_kwargs["activation_fn"], str
    ):
        policy_kwargs["activation_fn"] = getattr(
            torch.nn, policy_kwargs["activation_fn"]
        )

    valid_params = set(inspect.signature(PPO.__init__).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=verbose,
        **filtered,
    )


def make_vec_env(
    env_fns: Sequence[Callable[[], gym.Env]],
    *,
    use_subproc: bool = True,
) -> VecEnv:
    """Create a vectorized environment from factory callables.

    Args:
        env_fns: Sequence of zero-argument callables that create envs.
        use_subproc: Use ``SubprocVecEnv`` (True) or ``DummyVecEnv`` (False).

    Returns:
        A vectorized environment.
    """
    if use_subproc and len(env_fns) > 1:
        return SubprocVecEnv(list(env_fns))
    return DummyVecEnv(list(env_fns))
