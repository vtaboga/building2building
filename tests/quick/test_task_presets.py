"""Tests for building2building.config.tasks — task presets and resolution."""

from __future__ import annotations

import pytest

from building2building.config.tasks import (
    NORMALIZED_MODES,
    NORMALIZED_WEIGHT_LEVELS,
    TASK_PRESETS,
    TaskPreset,
    resolve_task_preset,
)
from building2building.types import (
    BarrierRewardConfig,
    DeadbandRewardConfig,
    NormalizedDeadbandRewardConfig,
)


@pytest.mark.quick
class TestTaskPresets:
    def test_paper_presets_exist(self) -> None:
        paper_tasks = {"task1", "task2", "task3", "task4", "task5"}
        assert paper_tasks.issubset(TASK_PRESETS.keys())

    def test_task3_legacy_preset_exists_for_ablation(self) -> None:
        assert "task3_legacy" in TASK_PRESETS

    @pytest.mark.parametrize(
        "name", ["task1", "task2", "task3", "task4", "task5", "task3_legacy"]
    )
    def test_preset_is_task_preset(self, name: str) -> None:
        assert isinstance(TASK_PRESETS[name], TaskPreset)

    def test_task1_deadband_low_energy_weight(self) -> None:
        p = TASK_PRESETS["task1"]
        assert isinstance(p.reward, DeadbandRewardConfig)
        assert p.reward.energy_weight == 0.01
        assert p.target_temperature_mode == "constant"
        assert p.target_temperature_occupied == 21.0
        assert p.target_temperature_unoccupied == 21.0
        assert p.unoccupied_policy == "fixed"

    def test_task2_deadband_high_energy_weight(self) -> None:
        p = TASK_PRESETS["task2"]
        assert isinstance(p.reward, DeadbandRewardConfig)
        assert p.reward.energy_weight == 0.10

    def test_task3_occupancy_mode_seasonal(self) -> None:
        p = TASK_PRESETS["task3"]
        assert p.target_temperature_mode == "occupancy"
        assert p.unoccupied_policy == "seasonal"
        assert p.seasonal_unoccupied_c is not None
        assert p.seasonal_unoccupied_c["winter"] == 18.0
        assert p.seasonal_unoccupied_c["shoulder"] == 21.0
        assert p.seasonal_unoccupied_c["summer"] == 26.0

    def test_task3_legacy_is_fixed(self) -> None:
        p = TASK_PRESETS["task3_legacy"]
        assert p.target_temperature_mode == "occupancy"
        assert p.unoccupied_policy == "fixed"
        assert p.target_temperature_unoccupied == 18.0

    def test_task4_barrier_reward(self) -> None:
        p = TASK_PRESETS["task4"]
        assert isinstance(p.reward, BarrierRewardConfig)
        assert p.reward.violation_penalty == 10.0

    def test_task5_random_schedule(self) -> None:
        p = TASK_PRESETS["task5"]
        assert isinstance(p.reward, DeadbandRewardConfig)
        assert p.target_temperature_mode == "random_schedule"


@pytest.mark.quick
class TestResolveTaskPreset:
    @pytest.mark.parametrize(
        "name", ["task1", "task2", "task3", "task4", "task5", "task3_legacy"]
    )
    def test_known_presets_resolve(self, name: str) -> None:
        preset = resolve_task_preset(name)
        assert preset is TASK_PRESETS[name]

    def test_unknown_preset_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown task preset"):
            resolve_task_preset("task99")


_NORMALIZED_NAMES: list[str] = [
    f"task_{m}_{w}"
    for m in NORMALIZED_MODES.keys()
    for w in NORMALIZED_WEIGHT_LEVELS.keys()
]


@pytest.mark.quick
class TestNormalizedTaskFamily:
    def test_nine_normalized_presets_exist(self) -> None:
        for name in _NORMALIZED_NAMES:
            assert name in TASK_PRESETS, f"missing normalized preset {name!r}"

    @pytest.mark.parametrize("name", _NORMALIZED_NAMES)
    def test_normalized_preset_uses_unfilled_normalized_config(
        self, name: str
    ) -> None:
        preset = TASK_PRESETS[name]
        assert isinstance(preset.reward, NormalizedDeadbandRewardConfig)
        # Presets must store the unfilled sentinel state; new_make_env
        # is responsible for resolving (tau_T, tau_E) at env-build time.
        assert not preset.reward.is_filled
        assert preset.reward.tau_T is None
        assert preset.reward.tau_E is None
        assert preset.reward.dT == 1.0

    @pytest.mark.parametrize(
        "name,expected_w",
        [
            ("task_const_w0", 0.0),
            ("task_const_wmed", 1.0),
            ("task_const_whigh", 5.0),
            ("task_occ_w0", 0.0),
            ("task_occ_wmed", 1.0),
            ("task_occ_whigh", 5.0),
            ("task_rand_w0", 0.0),
            ("task_rand_wmed", 1.0),
            ("task_rand_whigh", 5.0),
        ],
    )
    def test_normalized_preset_energy_weight(
        self, name: str, expected_w: float
    ) -> None:
        preset = TASK_PRESETS[name]
        assert isinstance(preset.reward, NormalizedDeadbandRewardConfig)
        assert preset.reward.energy_weight == expected_w

    @pytest.mark.parametrize(
        "name,expected_mode",
        [
            ("task_const_w0", "constant"),
            ("task_const_wmed", "constant"),
            ("task_const_whigh", "constant"),
            ("task_occ_w0", "occupancy"),
            ("task_occ_wmed", "occupancy"),
            ("task_occ_whigh", "occupancy"),
            ("task_rand_w0", "random_schedule"),
            ("task_rand_wmed", "random_schedule"),
            ("task_rand_whigh", "random_schedule"),
        ],
    )
    def test_normalized_preset_setpoint_mode(
        self, name: str, expected_mode: str
    ) -> None:
        preset = TASK_PRESETS[name]
        assert preset.target_temperature_mode == expected_mode

    def test_occupancy_presets_use_seasonal_unoccupied_policy(self) -> None:
        # The calibration regime is occupancy-based with the seasonal
        # winter-18 / shoulder-21 / summer-26 schedule; the three
        # ``task_occ_*`` presets must reproduce that exactly.
        for name in ("task_occ_w0", "task_occ_wmed", "task_occ_whigh"):
            preset = TASK_PRESETS[name]
            assert preset.unoccupied_policy == "seasonal"
            assert preset.seasonal_unoccupied_c is not None
            assert preset.seasonal_unoccupied_c["winter"] == 18.0
            assert preset.seasonal_unoccupied_c["shoulder"] == 21.0
            assert preset.seasonal_unoccupied_c["summer"] == 26.0

    def test_legacy_task1_unchanged(self) -> None:
        # Backward-compat invariant: task1 must keep its exact legacy
        # definition so existing PPO results / baseline_returns rows
        # remain reproducible.
        p = TASK_PRESETS["task1"]
        assert isinstance(p.reward, DeadbandRewardConfig)
        assert p.reward.energy_weight == 0.01
        assert p.reward.dT == 1.0
        assert p.target_temperature_mode == "constant"
        assert p.target_temperature_occupied == 21.0
        assert p.target_temperature_unoccupied == 21.0


@pytest.mark.quick
class TestMakeNormalizedDeadbandTaskFactory:
    """``make_normalized_deadband_task`` is a thin wrapper.

    It must lazily import the loader (so ``import
    building2building.config.tasks`` doesn't pull in dataset I/O), and
    when invoked it must produce a *filled* config.  We can't test the
    happy path here without committing reward_normalizers.yaml or
    monkeypatching, so we exercise the lazy-import + error-path
    contract.
    """

    def test_factory_lazy_imports_loader(self) -> None:
        # The import statement inside the factory body is intentional;
        # checking it's not a top-level import keeps test_task_presets
        # cheap (no metadata.parquet download at import time).
        import building2building.config.tasks as tasks_mod

        src = tasks_mod.__file__
        assert src is not None
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        # The loader import must live inside make_normalized_deadband_task
        # rather than at module top-level.
        assert (
            "from building2building.data.reward_normalizers import"
            in content
        )
        # Specifically, the import is *not* at file scope.
        head = content.split("def make_normalized_deadband_task", 1)[0]
        assert (
            "from building2building.data.reward_normalizers"
            not in head
        )
