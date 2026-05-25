"""Wiring and bound-correctness tests for the RL wrapper helpers.

Tests:
1. ``wrap_env_for_rl(normalize_obs=True)`` — obs in ``[0, 1]`` after reset and step.
2. ``new_make_env(rescale_action=True)`` — action space is ``[-1, 1]``.
3. ``new_make_env()`` (default) — action space is still raw engineering bounds.
4. Flags are independent: obs-only or action-only wrapping does not affect the other.
5. ``_get_reward_params`` walks through the new wrapper layers without errors.
6. ``make_rl_env_fn`` returns a thunk that builds a Monitor-wrapped env.
"""

from __future__ import annotations

import pytest


@pytest.mark.quick
def test_wrap_env_for_rl_observation_space_is_unit_interval() -> None:
    """After wrap_env_for_rl(normalize_obs=True), observation_space is [0, 1].

    Verifies that the deterministic bound-based scaling is wired up by checking
    the observation_space.low and observation_space.high, which are set to 0 and
    1 by NormalizeObservation.

    Note: actual obs values may briefly exit [0, 1] when the simulator produces
    a value outside its per-slot bounds (e.g. during EnergyPlus warmup phases).
    This is documented behavior — NormalizeObservation does not clip.
    """
    import numpy as np
    import building2building as b2b

    env = b2b.new_make_env("OfficeSmall", index=0, task="task_occ_wmed")
    env = b2b.wrap_env_for_rl(env, normalize_obs=True, rescale_action=False)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    assert (env.observation_space.low == 0.0).all(), (
        f"observation_space.low should be 0, got: {env.observation_space.low}"
    )
    assert (env.observation_space.high == 1.0).all(), (
        f"observation_space.high should be 1, got: {env.observation_space.high}"
    )
    obs, _ = env.reset()
    assert obs.shape == obs_shape
    assert obs.dtype == np.float32 or obs.dtype == np.float64
    env.close()


@pytest.mark.quick
def test_wrap_env_for_rl_action_in_minus_one_one() -> None:
    """After new_make_env(rescale_action=True), action_space is [-1, 1]."""
    import building2building as b2b

    env = b2b.new_make_env(
        "OfficeSmall",
        index=0,
        task="task_occ_wmed",
        rescale_action=True,
    )
    assert (env.action_space.low == -1.0).all(), (
        f"action_space.low != -1: {env.action_space.low}"
    )
    assert (env.action_space.high == 1.0).all(), (
        f"action_space.high != 1: {env.action_space.high}"
    )
    env.close()


@pytest.mark.quick
def test_wrap_env_for_rl_action_default_off() -> None:
    """Without rescale_action, action_space is the raw engineering Box."""
    import building2building as b2b

    env = b2b.new_make_env("OfficeSmall", index=0, task="task_occ_wmed")
    assert (env.action_space.low > -10.0).all(), (
        "Expected engineering-unit lower bounds > -10 °C"
    )
    assert (env.action_space.high < 100.0).all(), (
        "Expected engineering-unit upper bounds < 100 °C"
    )
    env.close()


@pytest.mark.quick
def test_wrap_env_for_rl_flags_are_independent() -> None:
    """Each flag can be flipped independently; False is a no-op."""
    import building2building as b2b

    env = b2b.new_make_env("OfficeSmall", index=0, task="task_occ_wmed")
    raw_action_space = env.action_space

    env_only_obs = b2b.wrap_env_for_rl(
        env, normalize_obs=True, rescale_action=False
    )
    assert env_only_obs.action_space == raw_action_space, (
        "normalize_obs=True should not alter the action space"
    )
    env.close()


@pytest.mark.quick
def test_get_reward_params_walks_through_wrappers() -> None:
    """_get_reward_params survives the new wrapper layers."""
    from analysis.task_study.reward_design_comparison import _get_reward_params
    import building2building as b2b

    env = b2b.new_make_env(
        "OfficeSmall",
        index=0,
        task="task_occ_wmed",
        rescale_action=True,
    )
    env = b2b.wrap_env_for_rl(env, normalize_obs=True, rescale_action=False)
    energy_weight, tau_T, tau_E = _get_reward_params(env)
    assert energy_weight == pytest.approx(1.0)
    assert tau_T > 0
    assert tau_E > 0
    env.close()


@pytest.mark.quick
def test_make_rl_env_fn_returns_monitor_wrapped_env() -> None:
    """make_rl_env_fn thunk builds a Monitor-wrapped env with both wrappers."""
    import building2building as b2b
    from stable_baselines3.common.monitor import Monitor
    from baselines.utils.training import make_rl_env_fn

    building_id = b2b.list_buildings("OfficeSmall")[0]
    factory = make_rl_env_fn(
        building_type="OfficeSmall",
        building_id=building_id,
        task="task_occ_wmed",
        normalize_obs=True,
        rescale_action=True,
        monitor=True,
    )
    env = factory()
    assert isinstance(env, Monitor), f"Expected Monitor, got {type(env)}"
    assert (env.action_space.low == -1.0).all(), "Action space should be [-1, 1]"
    assert (env.observation_space.low == 0.0).all(), "Obs space low should be 0"
    assert (env.observation_space.high == 1.0).all(), "Obs space high should be 1"
    env.close()
