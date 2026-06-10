"""Action-space transfer benchmark (Section 4, Table: Action-space transfer).

Building dynamics and reward stay fixed; controllable actuators change
between training and test.

Paper specification::

    System type      Training control       Test control          Dim change
    ─────────────────────────────────────────────────────────────────────────
    Unitary          Air flow rate          Air flow + SAT        5  → 10
    Central          VAV boxes only         VAV boxes + central   30 → 33
    Unitary          Air flow + SAT         Air flow rate         10 → 5
    Central          VAV boxes + central    VAV boxes only        33 → 30
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Literal, Sequence

import gymnasium as gym

from building2building.benchmarks.base import BenchmarkProblem
from building2building.pipeline.actuators import AnyEquipment, UnitarySystem, VAVSystem

SplitName = Literal["train", "test", "test_small"]

_DEFAULT_BUILDING_TYPE: dict[str, str] = {
    "unitary": "OfficeSmall",
    "central": "OfficeMedium",
}

_DEFAULT_UNITARY_SAT_VALUE = 22.0
_DEFAULT_CENTRAL_SAT_VALUE = 13.0


def _unitary_sat_overrides(
    equipment: Sequence[AnyEquipment],
    fixed_value: float = _DEFAULT_UNITARY_SAT_VALUE,
) -> dict[str, float]:
    """Return fixed-value overrides for SAT actuators in unitary systems.

    Each :class:`UnitarySystem` has two actuators per zone: fan air mass
    flow rate and supply air temperature setpoint.  The SAT actuators are
    identified by ``units == "Temperature"``.
    """
    overrides: dict[str, float] = {}
    for eq in equipment:
        if not isinstance(eq, UnitarySystem):
            continue
        for act in eq.actuators:
            if act.units == "Temperature":
                overrides[act.component_name] = fixed_value
    return overrides


def _central_sat_overrides(
    equipment: Sequence[AnyEquipment],
    fixed_value: float = _DEFAULT_CENTRAL_SAT_VALUE,
) -> dict[str, float]:
    """Return fixed-value overrides for central SAT actuators in VAV systems.

    Each :class:`VAVSystem` has one ``supply_temp_setpoint`` actuator that
    controls the central supply air temperature for the air loop.
    """
    overrides: dict[str, float] = {}
    for eq in equipment:
        if not isinstance(eq, VAVSystem):
            continue
        overrides[eq.supply_temp_setpoint.component_name] = fixed_value
    return overrides


class ActionSpaceTransfer(BenchmarkProblem):
    """Action-space transfer benchmark.

    The agent trains on a building with one set of controllable
    actuators and is tested on the **same building** with a different
    (expanded or reduced) actuator set.  Building dynamics and reward
    are identical between train and test.

    Args:
        system_type: HVAC system type — ``"unitary"`` targets
            :class:`~building2building.pipeline.actuators.UnitarySystem`
            buildings (default: OfficeSmall, 5 zones) where the
            reduced action space removes SAT setpoints;
            ``"central"`` targets
            :class:`~building2building.pipeline.actuators.VAVSystem`
            buildings (default: OfficeMedium, 15 zones) where the
            reduced action space removes the central supply-air
            temperature actuator.
        direction: ``"expand"`` means training uses the reduced set and
            testing uses the full set.  ``"reduce"`` is the reverse.
        task: Named task preset (e.g. ``"task_const_e0"``).
        building_type: Override the default building type for
            *system_type*.  If ``None``, uses ``"OfficeSmall"`` for
            unitary and ``"OfficeMedium"`` for central.
        split: Dataset split from which to select the building.
        split_index: Index within *split* to select the building.
    """

    def __init__(
        self,
        system_type: Literal["unitary", "central"] = "unitary",
        direction: Literal["expand", "reduce"] = "expand",
        task: str = "task_const_e0",
        building_type: str | None = None,
        split: SplitName = "train",
        split_index: int = 0,
    ) -> None:
        self.system_type = system_type
        self.direction = direction
        self.task = task
        self.building_type = building_type or _DEFAULT_BUILDING_TYPE[system_type]
        self.split = split
        self.split_index = split_index

    def _compute_overrides(
        self, equipment: Sequence[AnyEquipment], reduced: bool
    ) -> dict[str, float]:
        """Return actuator overrides for the given side.

        When *reduced* is ``False`` the full action space is used
        (empty overrides).  When ``True``, the appropriate actuators are
        pinned at their default operating values.
        """
        if not reduced:
            return {}
        if self.system_type == "unitary":
            return _unitary_sat_overrides(equipment)
        return _central_sat_overrides(equipment)

    def _make_env(self, reduced: bool, **kwargs: object) -> gym.Env:
        """Build a Gymnasium environment with full or reduced actuators.

        Mirrors the logic of
        :func:`~building2building.api.make_env` but injects
        ``fixed_actuator_overrides`` into the
        :class:`~building2building.types.BuildingConfig`.
        """
        from cattrs import structure

        from building2building.api import _resolve_effective_reward
        from building2building.config.tasks import resolve_task_preset
        from building2building.data.registry import BuildingInfo, get_registry
        from building2building.simulator import create_simulator
        from building2building.types import (
            BuildingConfig,
            RandomScheduleConfig,
            RunPeriodConfig,
            TaskConfig,
            ZoneTargetTemperatureConfig,
        )

        preset = resolve_task_preset(self.task)

        registry = get_registry()
        info: BuildingInfo = registry.get_building_by_index(
            self.building_type,  # type: ignore[arg-type]
            self.split,
            self.split_index,
        )
        effective_reward = _resolve_effective_reward(
            preset=preset,
            reward_override=None,
            building_type=self.building_type,
            building_id=info.building_id,
            run_period="full_year",
            normalizer_path=None,
        )

        eplus_output_dir = Path(tempfile.mkdtemp(prefix="b2b_eplus_"))
        eplus_output_dir.mkdir(parents=True, exist_ok=True)

        epjson_path = info.building_dir / "building.epjson"
        equipment_path = info.building_dir / "equipment.json"
        weather_path = info.building_dir / info.weather_file

        run_period_cfg = RunPeriodConfig.from_name("full_year")
        default_zone_target = ZoneTargetTemperatureConfig(
            occupied_c=preset.target_temperature_occupied,
            unoccupied_c=preset.target_temperature_unoccupied,
            unoccupied_policy=preset.unoccupied_policy,
            seasonal_unoccupied_c=(
                dict(preset.seasonal_unoccupied_c)
                if preset.seasonal_unoccupied_c is not None
                else None
            ),
        )
        random_schedule_cfg: RandomScheduleConfig | None = None
        if preset.target_temperature_mode == "random_schedule":
            random_schedule_cfg = RandomScheduleConfig(
                building_type=self.building_type,
                seed=0,
            )
        task_cfg = TaskConfig(
            run_period=run_period_cfg,
            target_temperature_mode=preset.target_temperature_mode,
            default_zone_target_temperature=default_zone_target,
            random_schedule_config=random_schedule_cfg,
        )

        equipment_data: list[AnyEquipment] = structure(
            json.loads(equipment_path.read_text()), list[AnyEquipment]
        )

        overrides = self._compute_overrides(equipment_data, reduced=reduced)

        building_config = BuildingConfig(
            path_to_building=epjson_path,
            path_to_weather=weather_path,
            reward_config=effective_reward,
            eplus_output_dir=eplus_output_dir,
            warmup_phases=info.warmup_phases,
            area=info.net_conditioned_area_m2,
            hvac_equipment=equipment_data,
            task_config=task_cfg,
            fixed_actuator_overrides=overrides,
        )

        env = create_simulator(building_config)
        steps = task_cfg.expected_steps()
        return gym.wrappers.TimeLimit(env, max_episode_steps=int(steps))

    def make_train_env(self, **kwargs: object) -> gym.Env:
        """Create a single training environment."""
        train_is_reduced = self.direction == "expand"
        return self._make_env(reduced=train_is_reduced, **kwargs)

    def make_test_env(self, **kwargs: object) -> gym.Env:
        """Create a single test environment."""
        test_is_reduced = self.direction == "reduce"
        return self._make_env(reduced=test_is_reduced, **kwargs)

    def make_train_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create training environments (one by default)."""
        return [self.make_train_env() for _ in range(n or 1)]

    def make_test_envs(self, n: int | None = None) -> list[gym.Env]:
        """Create test environments (one by default)."""
        return [self.make_test_env() for _ in range(n or 1)]
