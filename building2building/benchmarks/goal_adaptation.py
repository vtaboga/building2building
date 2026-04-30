"""Goal adaptation benchmark (Section 4, Table: Goal adaptation).

Training and test environments use the same building and control
interface, but the reward function / task changes between them.

Three canonical transfer axes ride on top of the normalized 3×3 task
family (:mod:`building2building.config.tasks`):

* **Trade-off transfer** (orthogonal to setpoint mode): train on
  ``task_occ_wmed``, evaluate on ``task_occ_w0`` or
  ``task_occ_whigh``.  All three are in the calibration regime
  (``mode="occupancy"``, ``dT=1.0``), so no calibration-mismatch
  warning fires; the benchmark isolates how a policy generalizes
  across ``w_E``.
* **Setpoint-mode transfer** (orthogonal to ``w_E``): train on
  ``task_occ_wmed``, evaluate on ``task_const_wmed`` or
  ``task_rand_wmed``.  The two test tasks are *outside* the
  calibration regime, so the simulator emits a one-time
  :class:`RuntimeWarning` at env construction.  This benchmark is
  the quantitative measurement of how much the approximate
  calibration costs in policy performance.
* **Legacy-vs-new transfer** (sanity check): train on ``task1``
  (un-normalized :class:`~building2building.types.DeadbandRewardConfig`),
  evaluate on ``task_const_wmed``.  Useful for asking whether a
  PPO checkpoint trained under the legacy reward generalizes to the
  new normalized reward "for free".

The constructor's defaults (``train_task="task_occ_wmed"``,
``test_task="task_occ_whigh"``) implement the trade-off-transfer axis.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from building2building.benchmarks.base import BenchmarkProblem
from building2building.data.download import BuildingType


class GoalAdaptation(BenchmarkProblem):
    """Goal adaptation benchmark.

    The agent trains with one task (reward / temperature target) and is
    tested with a different task on the *same* building.

    Args:
        building_type: Building type to use.
        split_index: Index within the train split.
        train_task: Named task preset for training.  Defaults to
            ``"task_occ_wmed"`` (balanced trade-off in the calibration
            regime).
        test_task: Named task preset for testing.  Defaults to
            ``"task_occ_whigh"`` (energy-emphasis trade-off in the
            calibration regime) — the "trade-off transfer" axis.  See
            module docstring for the other two canonical axes.
        run_period: Simulation run period.
    """

    def __init__(
        self,
        building_type: BuildingType = "OfficeSmall",
        split_index: int = 0,
        train_task: str = "task_occ_wmed",
        test_task: str = "task_occ_whigh",
        run_period: str = "full_year",
    ) -> None:
        self.building_type = building_type
        self.split_index = split_index
        self.train_task = train_task
        self.test_task = test_task
        self.run_period = run_period

    def make_train_env(self, **kwargs: object) -> gym.Env:
        """Create a single training environment."""
        from building2building.api import new_make_env

        return new_make_env(
            building_type=self.building_type,
            split="train",
            index=self.split_index,
            task=self.train_task,
            run_period=self.run_period,
            **kwargs,  # type: ignore[arg-type]
        )

    def make_test_env(self, **kwargs: object) -> gym.Env:
        """Create a single test environment."""
        from building2building.api import new_make_env

        return new_make_env(
            building_type=self.building_type,
            split="train",
            index=self.split_index,
            task=self.test_task,
            run_period=self.run_period,
            **kwargs,  # type: ignore[arg-type]
        )

    def make_train_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create training environments (one by default)."""
        return [self.make_train_env() for _ in range(n or 1)]

    def make_test_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create test environments (one by default)."""
        return [self.make_test_env() for _ in range(n or 1)]
