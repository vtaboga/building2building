"""Guard the ``new_make_env`` regression: preset mode must not be overridden.

Prior to the fix, ``new_make_env`` silently defaulted
``target_temperature_mode`` to ``"constant"``, which caused the
occupancy-based task3 (and later task5) to run in constant mode in
every baseline script.  These tests assert the new behaviour: when the
caller does not pass ``target_temperature_mode``, the preset's mode
wins.

The tests exercise a lightweight patched version of ``new_make_env``
that stops right after the ``TaskConfig`` is built, so they do **not**
require the HuggingFace registry or EnergyPlus.
"""
# This file pins the public API contract.
# Changes here = breaking API changes; requires a CHANGELOG.md entry.
# Marker applied automatically by conftest.py (api_contract glob).


from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import building2building.api as api_mod
from building2building.config.tasks import TASK_PRESETS, TaskPreset
from building2building.types import RewardConfig, TaskConfig


_FILLED_REWARD = RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0)


class _StubInfo:
    def __init__(self, tmp_path: Path) -> None:
        self.building_dir = tmp_path
        self.weather_file = "weather.epw"
        self.warmup_phases = 0
        self.net_conditioned_area_m2 = 100.0
        self.climate_zone = 5


class _StubRegistry:
    def __init__(self, info: _StubInfo) -> None:
        self._info = info

    def get_building_by_index(self, *_args: Any, **_kwargs: Any) -> _StubInfo:
        return self._info

    def get_building_by_id(self, *_args: Any, **_kwargs: Any) -> _StubInfo:
        return self._info


class _CapturedConfig(Exception):
    """Used to bail out of new_make_env once we have the TaskConfig."""

    def __init__(self, task: TaskConfig, preset: TaskPreset) -> None:
        super().__init__("captured")
        self.task = task
        self.preset = preset


def _patch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = _StubInfo(tmp_path)
    (tmp_path / "building.epjson").write_text("{}")
    (tmp_path / "equipment.json").write_text("[]")
    (tmp_path / "weather.epw").write_text("")

    from building2building.data import registry as registry_mod

    monkeypatch.setattr(registry_mod, "get_registry", lambda: _StubRegistry(info))

    def _raise(building_config: Any) -> Any:
        # building_config.task_config is the TaskConfig we want to inspect.
        raise _CapturedConfig(building_config.task_config, preset=None)  # type: ignore[arg-type]

    from building2building import simulator as simulator_mod

    monkeypatch.setattr(simulator_mod, "create_simulator", _raise)

    monkeypatch.setattr(api_mod, "_patch_epjson_run_period", lambda *a, **k: None)


@pytest.mark.quick
class TestNewMakeEnvModeDefault:
    @pytest.mark.parametrize(
        ("task_name", "expected_mode"),
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
    def test_preset_mode_wins_when_default(
        self,
        task_name: str,
        expected_mode: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_env(monkeypatch, tmp_path)
        with pytest.raises(_CapturedConfig) as excinfo:
            api_mod.new_make_env("OfficeSmall", task=task_name, reward=_FILLED_REWARD)
        assert excinfo.value.task.target_temperature_mode == expected_mode

    def test_explicit_mode_overrides_preset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_env(monkeypatch, tmp_path)
        with pytest.raises(_CapturedConfig) as excinfo:
            api_mod.new_make_env(
                "OfficeSmall",
                task="task_occ_emed",
                target_temperature_mode="constant",
                reward=_FILLED_REWARD,
            )
        assert excinfo.value.task.target_temperature_mode == "constant"

    def test_occ_task_carries_seasonal_unoccupied(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_env(monkeypatch, tmp_path)
        with pytest.raises(_CapturedConfig) as excinfo:
            api_mod.new_make_env("OfficeSmall", task="task_occ_emed", reward=_FILLED_REWARD)
        zone_target = excinfo.value.task.default_zone_target_temperature
        assert zone_target.unoccupied_policy == "seasonal"
        assert zone_target.seasonal_unoccupied_c is not None
        assert zone_target.seasonal_unoccupied_c["summer"] == 26.0

    def test_rand_task_populates_random_schedule_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_env(monkeypatch, tmp_path)
        with pytest.raises(_CapturedConfig) as excinfo:
            api_mod.new_make_env(
                "OfficeSmall",
                task="task_rand_emed",
                random_schedule_seed=42,
                reward=_FILLED_REWARD,
            )
        rs = excinfo.value.task.random_schedule_config
        assert rs is not None
        assert rs.building_type == "OfficeSmall"
        assert rs.seed == 42

    def test_const_task_has_no_random_schedule(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_env(monkeypatch, tmp_path)
        with pytest.raises(_CapturedConfig) as excinfo:
            api_mod.new_make_env("OfficeSmall", task="task_const_e0", reward=_FILLED_REWARD)
        assert excinfo.value.task.random_schedule_config is None

    def test_preset_used_intact(self) -> None:
        assert TASK_PRESETS["task_occ_emed"].target_temperature_mode == "occupancy"
        assert (
            TASK_PRESETS["task_rand_emed"].target_temperature_mode == "random_schedule"
        )
