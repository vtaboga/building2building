"""SB3 training utilities for the baselines."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor

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


PAPER_SAC_DEFAULTS: dict[str, Any] = {
    "learning_rate": 3.0e-4,
    "buffer_size": 300_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "policy_kwargs": {
        "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        "activation_fn": torch.nn.ReLU,
    },
}


def build_sac(
    env: VecEnv,
    *,
    tensorboard_log: str | None = None,
    seed: int | None = None,
    verbose: int = 1,
    **overrides: Any,
) -> SAC:
    """Create a SAC model with the B2B default hyperparameters.

    Any key from :data:`PAPER_SAC_DEFAULTS` can be overridden via
    keyword arguments.

    Args:
        env: Vectorized environment.  Observation normalisation is handled
            by the per-env :class:`~building2building.simulator.wrappers.NormalizeObservation`
            wrapper applied via :func:`make_rl_env_fn`; no ``VecNormalize`` is
            required.
        tensorboard_log: TensorBoard log directory.
        seed: Random seed.
        verbose: SB3 verbosity level (0=silent, 1=info, 2=debug).
        **overrides: SAC constructor keyword overrides.

    Returns:
        A configured :class:`SAC` instance.
    """
    kwargs = {**PAPER_SAC_DEFAULTS, **overrides}

    policy_kwargs = dict(kwargs.pop("policy_kwargs", {}))
    if "activation_fn" in policy_kwargs and isinstance(
        policy_kwargs["activation_fn"], str
    ):
        policy_kwargs["activation_fn"] = getattr(
            torch.nn, policy_kwargs["activation_fn"]
        )

    valid_params = set(inspect.signature(SAC.__init__).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return SAC(
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


def make_rl_env_fn(
    *,
    building_type: str,
    building_id: str | None,
    task: Any,
    run_period: str = "full_year",
    normalize_obs: bool = True,
    rescale_action: bool = True,
    monitor: bool = True,
    normalizer_path: Path | None = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a single wrapped env for an RL run.

    The thunk is what :class:`~stable_baselines3.common.vec_env.SubprocVecEnv`
    and :class:`~stable_baselines3.common.vec_env.DummyVecEnv` consume.

    The returned env is constructed via
    :func:`building2building.api.make_env` (with the
    ``rescale_action`` keyword), then passed through
    :func:`building2building.wrap_env_for_rl` (with ``normalize_obs``).
    Optionally wrapped in :class:`~stable_baselines3.common.monitor.Monitor`.

    All four of ``train_ppo``, ``train_sac``, ``tune_ppo``, ``eval_ppo``
    use this; the analysis modules use it too, ensuring the wrapper stack
    is identical across all RL code paths::

        Monitor(NormalizeObservation(TimeLimit(RescaleAction(simulator))))

    Note: ``RescaleAction`` is applied inside ``TimeLimit`` because
    ``make_env`` wraps with ``RescaleAction`` before returning the
    ``TimeLimit``-wrapped env.

    Args:
        building_type: Building type string (e.g. ``"OfficeSmall"``).
        building_id: Explicit building identifier.
        task: Task name or preset passed to :func:`building2building.api.make_env`.
        run_period: Simulation run period (``"full_year"``, ``"winter"``,
            ``"summer"``).
        normalize_obs: Apply deterministic ``[0, 1]`` observation scaling
            via :class:`~building2building.simulator.wrappers.NormalizeObservation`.
        rescale_action: Rescale the action space to ``[-1, 1]``.
            Applied via ``make_env(rescale_action=True)`` so that
            ``wrap_env_for_rl`` is always called with
            ``rescale_action=False`` to avoid double-rescaling.
        monitor: Wrap the env in
            :class:`~stable_baselines3.common.monitor.Monitor`.
        normalizer_path: Override the default reward-normalizer YAML
            passed to :func:`building2building.api.make_env`.

    Returns:
        A zero-argument callable that, when called, returns a fully
        wrapped :class:`gymnasium.Env`.
    """
    import building2building as b2b

    def _factory() -> gym.Env:
        env = b2b.make_env(
            building_type,
            building_id=building_id,
            task=task,
            run_period=run_period,
            rescale_action=rescale_action,
            normalizer_path=normalizer_path,
        )
        env = b2b.wrap_env_for_rl(
            env,
            normalize_obs=normalize_obs,
            rescale_action=False,
        )
        if monitor:
            env = Monitor(env)
        return env

    return _factory
