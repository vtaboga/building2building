"""Tests for building2building.config.tasks — task presets and resolution."""

from __future__ import annotations

import pytest

from building2building.config.tasks import TASK_PRESETS, TaskPreset, resolve_task_preset
from building2building.types import BarrierRewardConfig, DeadbandRewardConfig


@pytest.mark.quick
class TestTaskPresets:
    def test_all_four_presets_exist(self) -> None:
        assert set(TASK_PRESETS.keys()) == {"task1", "task2", "task3", "task4"}

    @pytest.mark.parametrize("name", ["task1", "task2", "task3", "task4"])
    def test_preset_is_task_preset(self, name: str) -> None:
        assert isinstance(TASK_PRESETS[name], TaskPreset)

    def test_task1_deadband_low_energy_weight(self) -> None:
        p = TASK_PRESETS["task1"]
        assert isinstance(p.reward, DeadbandRewardConfig)
        assert p.reward.energy_weight == 0.01
        assert p.target_temperature_mode == "constant"
        assert p.target_temperature_occupied == 21.0
        assert p.target_temperature_unoccupied == 21.0

    def test_task2_deadband_high_energy_weight(self) -> None:
        p = TASK_PRESETS["task2"]
        assert isinstance(p.reward, DeadbandRewardConfig)
        assert p.reward.energy_weight == 0.10

    def test_task3_occupancy_mode(self) -> None:
        p = TASK_PRESETS["task3"]
        assert p.target_temperature_mode == "occupancy"
        assert p.target_temperature_unoccupied == 18.0

    def test_task4_barrier_reward(self) -> None:
        p = TASK_PRESETS["task4"]
        assert isinstance(p.reward, BarrierRewardConfig)
        assert p.reward.violation_penalty == 10.0


@pytest.mark.quick
class TestResolveTaskPreset:
    @pytest.mark.parametrize("name", ["task1", "task2", "task3", "task4"])
    def test_known_presets_resolve(self, name: str) -> None:
        preset = resolve_task_preset(name)
        assert preset is TASK_PRESETS[name]

    def test_unknown_preset_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown task preset"):
            resolve_task_preset("task99")
