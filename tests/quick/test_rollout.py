"""Tests for :mod:`building2building.api.rollout`.

The ``to_npz`` / ``from_npz`` round-trip is a quick pure-Python test (no
EnergyPlus needed). Running a real rollout against an EnergyPlus env is
marked ``long`` and requires the dataset + EnergyPlus runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import building2building as b2b
import building2building.api as api_mod
from building2building.api.rollout import Trajectory, callable_controller
from building2building.config.tasks import TASK_PRESETS
from building2building.types import RunPeriodConfig


def _make_fake_trajectory(t: int = 5) -> Trajectory:
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(t + 1, 4)).astype(np.float64)
    actions = rng.normal(size=(t, 2)).astype(np.float64)
    rewards = rng.normal(size=(t,)).astype(np.float64)
    terminateds = np.zeros(t, dtype=bool)
    truncateds = np.zeros(t, dtype=bool)
    truncateds[-1] = True
    raw_obs = [
        {
            "temperature": {"zone_a": 20.0 + i, "zone_b": 21.5 + i},
            "energy": {"electricity": 100.0 * i, "natural_gas": 10.0 * i},
        }
        for i in range(t + 1)
    ]
    infos = [{"step": i} for i in range(t + 1)]
    return Trajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminateds=terminateds,
        truncateds=truncateds,
        infos=infos,
        raw_observations=raw_obs,
        controlled_zones=["zone_a", "zone_b"],
        task_config=None,
        building_info=None,
        observation_names=["t_a", "t_b", "e_elec", "e_gas"],
    )


@pytest.mark.quick
class TestTrajectoryRoundTrip:
    def test_length(self) -> None:
        traj = _make_fake_trajectory(t=5)
        assert len(traj) == 5

    def test_to_from_npz_preserves_arrays(self, tmp_path: Path) -> None:
        traj = _make_fake_trajectory(t=5)
        path = tmp_path / "traj.npz"
        traj.to_npz(path)
        assert path.exists()
        loaded = Trajectory.from_npz(path)
        np.testing.assert_array_equal(loaded.observations, traj.observations)
        np.testing.assert_array_equal(loaded.actions, traj.actions)
        np.testing.assert_array_equal(loaded.rewards, traj.rewards)
        np.testing.assert_array_equal(loaded.terminateds, traj.terminateds)
        np.testing.assert_array_equal(loaded.truncateds, traj.truncateds)

    def test_to_from_npz_preserves_nested(self, tmp_path: Path) -> None:
        traj = _make_fake_trajectory(t=3)
        path = tmp_path / "traj.npz"
        traj.to_npz(path)
        loaded = Trajectory.from_npz(path)
        assert loaded.infos == traj.infos
        assert loaded.raw_observations == traj.raw_observations
        assert loaded.controlled_zones == traj.controlled_zones
        assert loaded.observation_names == traj.observation_names

    def test_to_from_npz_preserves_real_nested_dataclasses(
        self,
        tmp_path: Path,
        fixture_registry: Any,
    ) -> None:
        building_info = fixture_registry.get_building_by_id("OfficeSmall", "fixture-0001")
        task_config = api_mod._resolve_task_config(
            preset=TASK_PRESETS["task_occ_emed"],
            run_period_cfg=RunPeriodConfig.from_name("winter"),
            timesteps_per_hour=12,
            target_temperature_mode=None,
            random_schedule_seed=123,
            building_type="OfficeSmall",
        )
        traj = _make_fake_trajectory(t=3)
        traj.building_info = building_info
        traj.task_config = task_config

        path = tmp_path / "traj_with_dataclasses.npz"
        traj.to_npz(path)
        loaded = Trajectory.from_npz(path)

        assert loaded.building_info == building_info
        assert loaded.task_config == task_config


@pytest.mark.quick
class TestControllerProtocol:
    def test_callable_controller_wraps_plain_fn(self) -> None:
        fn = lambda obs: np.asarray(obs) * 2.0  # noqa: E731
        ctrl = callable_controller(fn)
        ctrl.reset(env=None)  # type: ignore[arg-type]
        out = ctrl(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(out, np.array([2.0, 4.0]))

    def test_public_api_exports(self) -> None:
        assert hasattr(b2b, "rollout")
        assert hasattr(b2b, "Trajectory")
        assert hasattr(b2b, "Controller")
        assert hasattr(b2b, "callable_controller")


# ---------------------------------------------------------------------------
# Long test — needs EnergyPlus + the dataset on HuggingFace.
# ---------------------------------------------------------------------------


@pytest.mark.long
class TestRolloutEndToEnd:
    def test_small_rollout_captures_expected_shapes(self, tmp_path: Path) -> None:
        env = b2b.make_env(
            "OfficeSmall",
            task="task_const_e0",
            run_period="winter",
            max_episode_steps=5,
        )
        try:
            rng = np.random.default_rng(0)

            def _random(_obs):
                return env.action_space.sample()

            traj = b2b.rollout(env, _random, seed=0, max_steps=5)
            assert traj.observations.shape[0] == 6  # T+1
            assert traj.actions.shape[0] == 5
            assert traj.rewards.shape == (5,)
            assert traj.terminateds.shape == (5,)
            assert traj.truncateds.shape == (5,)
            assert len(traj.raw_observations) == 6

            path = tmp_path / "traj.npz"
            traj.to_npz(path)
            reloaded = Trajectory.from_npz(path)
            np.testing.assert_array_equal(reloaded.rewards, traj.rewards)

            assert traj.building_info is not None
            assert traj.task_config is not None
            # OfficeSmall has an ASHRAE climate zone assigned.
            assert isinstance(traj.building_info.climate_zone, int)
        finally:
            env.close()
