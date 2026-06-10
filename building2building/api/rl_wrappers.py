"""RL-specific environment wrappers for Building2Building.

Single source of truth for "wrap an env for RL training".  Both
``baselines/`` and ``analysis/`` import from here so the wrapper stack
is identical across all code paths.
"""

from __future__ import annotations

import gymnasium as gym

from building2building.simulator.wrappers import NormalizeObservation


def wrap_env_for_rl(
    env: gym.Env,
    *,
    normalize_obs: bool = True,
    rescale_action: bool = False,
) -> gym.Env:
    """Wrap a Building2Building Gymnasium env for RL training.

    Composes a small, well-understood wrapper stack:

    1. (optional) ``gym.wrappers.RescaleAction(env, -1.0, 1.0)`` —
       rescales the action ``Box`` to ``[-1, 1]`` per actuator. The
       wrapper handles the inverse mapping internally; the underlying
       env still receives engineering-unit actions.
    2. (optional) ``b2b.simulator.wrappers.NormalizeObservation(env)``
       — deterministic per-feature ``[0, 1]`` rescaling using
       ``observation_space.low`` and ``observation_space.high`` exposed
       by ``flat_observation_info``. *Not* running statistics.

    Both wrappers are optional and applied in this order so that the
    *outermost* observation space is ``Box([0, 1], ...)`` and the
    *outermost* action space is ``Box([-1, 1], ...)`` when both are
    enabled.

    Args:
        env: Gymnasium env (typically the result of
            :func:`building2building.api.make_env`).
        normalize_obs: Wrap with :class:`NormalizeObservation`.
            Defaults to ``True`` because RL policies almost always
            benefit from this; opt out for analysis tools that need
            raw observations.
        rescale_action: Wrap with :class:`gym.wrappers.RescaleAction`
            so the agent emits actions in ``[-1, 1]``. Defaults to
            ``False`` because non-RL consumers (reactive controllers,
            benchmark harnesses, manual rollouts) emit raw setpoints
            in engineering units. RL training/eval scripts must pass
            ``rescale_action=True``.

    Returns:
        The wrapped env. Inner ``env.metadata`` is preserved (the
        wrappers chain via ``getattr`` fallthrough).

    Notes:
        Order matters: ``RescaleAction`` is applied first (innermost)
        so that the policy's outermost action space is the rescaled
        ``[-1, 1]`` space. ``NormalizeObservation`` is applied second
        (outermost) so the policy's outermost observation space is
        ``[0, 1]``.

        Observations may *briefly* exit ``[0, 1]`` if the simulator
        produces a value outside the per-slot bounds (e.g. zone temp
        > 45°C). This is by design — the ``NormalizeObservation``
        wrapper does not clip, it just affine-maps. SB3's policies
        handle this gracefully.

        If ``rescale_action=True`` was already applied via
        ``make_env(rescale_action=True)``, do **not** pass
        ``rescale_action=True`` here as well — that would add a second
        ``RescaleAction`` layer (which is a no-op but needlessly
        deepens the chain).  The :func:`make_rl_env_fn` helper in
        ``baselines.utils.training`` handles this correctly by always
        passing ``rescale_action=False`` to this function.
    """
    if rescale_action:
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    if normalize_obs:
        env = NormalizeObservation(env)
    return env


__all__ = ["wrap_env_for_rl"]
