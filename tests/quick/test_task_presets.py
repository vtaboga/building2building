"""Tests for building2building.config.tasks — task presets and resolution."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from building2building.config.tasks import (
    NORMALIZED_MODES,
    NORMALIZED_WEIGHT_LEVELS,
    TASK_PRESETS,
    TaskPreset,
    resolve_task_preset,
)
from building2building.types import NormalizedDeadbandRewardConfig

_GRID_NAMES: list[str] = [
    f"task_{m}_{w}"
    for m in NORMALIZED_MODES.keys()
    for w in NORMALIZED_WEIGHT_LEVELS.keys()
]


@pytest.mark.quick
class TestNineTaskGrid:
    def test_nine_presets_exist(self) -> None:
        for name in _GRID_NAMES:
            assert name in TASK_PRESETS, f"missing preset {name!r}"

    @pytest.mark.parametrize("name", _GRID_NAMES)
    def test_preset_is_task_preset_instance(self, name: str) -> None:
        assert isinstance(TASK_PRESETS[name], TaskPreset)

    @pytest.mark.parametrize("name", _GRID_NAMES)
    def test_preset_uses_unfilled_normalized_config(self, name: str) -> None:
        preset = TASK_PRESETS[name]
        assert isinstance(preset.reward, NormalizedDeadbandRewardConfig)
        # Unfilled sentinel: tau_T/tau_E resolved at env-build time.
        assert not preset.reward.is_filled
        assert preset.reward.dT == 1.0

    @pytest.mark.parametrize(
        "name,expected_weight",
        [
            ("task_const_e0", 0.0),
            ("task_const_emed", 1.0),
            ("task_const_ehigh", 5.0),
            ("task_occ_e0", 0.0),
            ("task_occ_emed", 1.0),
            ("task_occ_ehigh", 5.0),
            ("task_rand_e0", 0.0),
            ("task_rand_emed", 1.0),
            ("task_rand_ehigh", 5.0),
        ],
    )
    def test_energy_weight(self, name: str, expected_weight: float) -> None:
        preset = TASK_PRESETS[name]
        assert isinstance(preset.reward, NormalizedDeadbandRewardConfig)
        assert preset.reward.energy_weight == expected_weight

    @pytest.mark.parametrize(
        "name,expected_mode",
        [
            ("task_const_e0", "constant"),
            ("task_const_emed", "constant"),
            ("task_const_ehigh", "constant"),
            ("task_occ_e0", "occupancy"),
            ("task_occ_emed", "occupancy"),
            ("task_occ_ehigh", "occupancy"),
            ("task_rand_e0", "random_schedule"),
            ("task_rand_emed", "random_schedule"),
            ("task_rand_ehigh", "random_schedule"),
        ],
    )
    def test_setpoint_mode(self, name: str, expected_mode: str) -> None:
        preset = TASK_PRESETS[name]
        assert preset.target_temperature_mode == expected_mode

    def test_occ_presets_use_seasonal_unoccupied_policy(self) -> None:
        for name in ("task_occ_e0", "task_occ_emed", "task_occ_ehigh"):
            preset = TASK_PRESETS[name]
            assert preset.unoccupied_policy == "seasonal"
            assert preset.seasonal_unoccupied_c is not None
            assert preset.seasonal_unoccupied_c["winter"] == 18.0
            assert preset.seasonal_unoccupied_c["shoulder"] == 21.0
            assert preset.seasonal_unoccupied_c["summer"] == 26.0


@pytest.mark.quick
class TestResolveTaskPreset:
    @pytest.mark.parametrize("name", _GRID_NAMES)
    def test_known_presets_resolve(self, name: str) -> None:
        preset = resolve_task_preset(name)
        assert preset is TASK_PRESETS[name]

    def test_unknown_preset_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown task preset"):
            resolve_task_preset("task99")


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
        code = textwrap.dedent(
            """
            import sys
            import building2building.config.tasks  # noqa: F401
            assert (
                "building2building.data.reward_normalizers" not in sys.modules
            ), sorted(m for m in sys.modules if m.startswith("building2building"))
            """
        )
        subprocess.run([sys.executable, "-c", code], check=True)
