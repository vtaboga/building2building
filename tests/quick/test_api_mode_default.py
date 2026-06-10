"""Tests for mode and reward resolution helpers in ``building2building.api``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import building2building.api as api_mod
from building2building.config.tasks import TASK_PRESETS, TaskPreset
from building2building.types import (
    NormalizedDeadbandRewardConfig,
    RewardConfig,
    RunPeriodConfig,
)

_FILLED_REWARD = RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0)
_REWARD_NORMALIZERS_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "reward_normalizers_fixture.yaml"
)


def _patch_registry(monkeypatch: pytest.MonkeyPatch, fixture_registry: Any) -> None:
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )


@pytest.mark.quick
class TestTaskConfigResolution:
    @pytest.mark.parametrize("task_name", sorted(TASK_PRESETS))
    def test_preset_mode_wins_when_default(self, task_name: str) -> None:
        preset = TASK_PRESETS[task_name]
        cfg = api_mod._resolve_task_config(
            preset=preset,
            run_period_cfg=RunPeriodConfig.from_name("full_year"),
            timesteps_per_hour=12,
            target_temperature_mode=None,
            random_schedule_seed=42,
            building_type="OfficeSmall",
        )
        assert cfg.target_temperature_mode == preset.target_temperature_mode
        assert cfg.default_zone_target_temperature.occupied_c == pytest.approx(
            preset.target_temperature_occupied
        )
        assert cfg.default_zone_target_temperature.unoccupied_c == pytest.approx(
            preset.target_temperature_unoccupied
        )
        if preset.target_temperature_mode == "random_schedule":
            assert cfg.random_schedule_config is not None
            assert cfg.random_schedule_config.building_type == "OfficeSmall"
            assert cfg.random_schedule_config.seed == 42
        else:
            assert cfg.random_schedule_config is None

    def test_explicit_mode_overrides_preset(self) -> None:
        cfg = api_mod._resolve_task_config(
            preset=TASK_PRESETS["task_occ_emed"],
            run_period_cfg=RunPeriodConfig.from_name("full_year"),
            timesteps_per_hour=12,
            target_temperature_mode="constant",
            random_schedule_seed=42,
            building_type="OfficeSmall",
        )
        assert cfg.target_temperature_mode == "constant"
        assert cfg.random_schedule_config is None


@pytest.mark.quick
class TestRewardResolution:
    def test_non_normalized_preset_reward_returns_unchanged(self) -> None:
        custom_reward = RewardConfig(energy_weight=2.0, dT=1.5, tau_T=1.2, tau_E=0.8)
        custom_preset = TaskPreset(
            reward=custom_reward,
            target_temperature_mode="constant",
            target_temperature_occupied=21.0,
            target_temperature_unoccupied=21.0,
        )
        resolved = api_mod._resolve_effective_reward(
            preset=custom_preset,
            reward_override=None,
            building_type="OfficeSmall",
            building_id="fixture-0001",
            run_period="full_year",
            normalizer_path=_REWARD_NORMALIZERS_FIXTURE,
        )
        assert resolved is custom_reward

    def test_explicit_reward_override_wins(self) -> None:
        resolved = api_mod._resolve_effective_reward(
            preset=TASK_PRESETS["task_occ_emed"],
            reward_override=_FILLED_REWARD,
            building_type="OfficeSmall",
            building_id="fixture-0001",
            run_period="full_year",
            normalizer_path=_REWARD_NORMALIZERS_FIXTURE,
        )
        assert resolved is _FILLED_REWARD

    def test_normalized_preset_is_filled(
        self, monkeypatch: pytest.MonkeyPatch, fixture_registry: Any
    ) -> None:
        _patch_registry(monkeypatch, fixture_registry)
        resolved = api_mod._resolve_effective_reward(
            preset=TASK_PRESETS["task_occ_emed"],
            reward_override=None,
            building_type="OfficeSmall",
            building_id="fixture-0001",
            run_period="full_year",
            normalizer_path=_REWARD_NORMALIZERS_FIXTURE,
        )
        assert isinstance(resolved, NormalizedDeadbandRewardConfig)
        assert resolved.is_filled
        assert resolved.tau_T == pytest.approx(2.0)
        assert resolved.tau_E == pytest.approx(3.0)

    def test_stale_yaml_missing_bucket_raises_key_error(
        self, monkeypatch: pytest.MonkeyPatch, fixture_registry: Any
    ) -> None:
        _patch_registry(monkeypatch, fixture_registry)
        with pytest.raises(KeyError):
            api_mod._resolve_effective_reward(
                preset=TASK_PRESETS["task_occ_emed"],
                reward_override=None,
                building_type="OfficeMedium",
                building_id="fixture-0001",
                run_period="full_year",
                normalizer_path=_REWARD_NORMALIZERS_FIXTURE,
            )


@pytest.mark.quick
def test_make_env_mode_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
) -> None:
    _patch_registry(monkeypatch, fixture_registry)
    env = api_mod.make_env(
        "OfficeSmall",
        task="task_occ_emed",
        reward=_FILLED_REWARD,
        max_episode_steps=4,
    )
    try:
        assert env.unwrapped.metadata["task_config"].target_temperature_mode == "occupancy"
    finally:
        env.close()
