"""Tests for building2building.types — run period, task, and reward configs."""
# This file pins the public API contract.
# Changes here = breaking API changes; requires a CHANGELOG.md entry.
# Marker applied automatically by conftest.py (api_contract glob).


from __future__ import annotations

import pytest

from building2building.types import (
    NormalizedDeadbandRewardConfig,
    RandomScheduleConfig,
    RunPeriodConfig,
    TaskConfig,
    ZoneTargetTemperatureConfig,
    reward_config_from_dict,
)


@pytest.mark.quick
class TestRunPeriodConfig:
    @pytest.mark.parametrize("name", ["full_year", "winter", "summer"])
    def test_from_name_valid(self, name: str) -> None:
        rp = RunPeriodConfig.from_name(name)
        assert rp.name == name
        assert rp.begin_month >= 1
        assert rp.end_month <= 12

    def test_from_name_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            RunPeriodConfig.from_name("autumn")

    def test_full_year_expected_steps(self) -> None:
        rp = RunPeriodConfig.from_name("full_year")
        assert rp.expected_steps(12) == 365 * 24 * 12

    def test_winter_expected_steps(self) -> None:
        rp = RunPeriodConfig.from_name("winter")
        assert rp.expected_steps(4) == 90 * 24 * 4

    def test_expected_steps_zero_raises(self) -> None:
        rp = RunPeriodConfig.from_name("full_year")
        with pytest.raises(ValueError, match="must be > 0"):
            rp.expected_steps(0)


@pytest.mark.quick
class TestZoneTargetTemperatureConfig:
    def test_from_dict_with_values(self) -> None:
        cfg = ZoneTargetTemperatureConfig.from_dict(
            {"occupied_c": 22.0, "unoccupied_c": 18.0},
            fallback_temperature_c=21.0,
        )
        assert cfg.occupied_c == 22.0
        assert cfg.unoccupied_c == 18.0

    def test_from_dict_uses_fallback(self) -> None:
        cfg = ZoneTargetTemperatureConfig.from_dict({}, fallback_temperature_c=19.0)
        assert cfg.occupied_c == 19.0
        assert cfg.unoccupied_c == 19.0

    def test_unoccupied_defaults_to_occupied(self) -> None:
        cfg = ZoneTargetTemperatureConfig.from_dict(
            {"occupied_c": 23.0}, fallback_temperature_c=21.0
        )
        assert cfg.unoccupied_c == 23.0


@pytest.mark.quick
class TestTaskConfig:
    def test_from_dict_defaults(self) -> None:
        tc = TaskConfig.from_dict({})
        assert tc.run_period.name == "full_year"
        assert tc.target_temperature_mode == "constant"
        assert tc.timesteps_per_hour == 12
        assert tc.default_zone_target_temperature.occupied_c == 21.0

    def test_from_dict_explicit_values(self) -> None:
        tc = TaskConfig.from_dict(
            {
                "run_period": "winter",
                "target_temperature_mode": "occupancy",
                "timesteps_per_hour": 4,
                "default_zone_target_temperature": {
                    "occupied_c": 22.0,
                    "unoccupied_c": 18.0,
                },
            }
        )
        assert tc.run_period.name == "winter"
        assert tc.target_temperature_mode == "occupancy"
        assert tc.timesteps_per_hour == 4

    def test_invalid_temperature_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="target_temperature_mode"):
            TaskConfig.from_dict({"target_temperature_mode": "magic"})

    def test_invalid_timesteps_per_hour_raises(self) -> None:
        with pytest.raises(ValueError, match="timesteps_per_hour"):
            TaskConfig.from_dict({"timesteps_per_hour": 7})

    @pytest.mark.parametrize("tph", [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60])
    def test_all_valid_timesteps_per_hour(self, tph: int) -> None:
        tc = TaskConfig.from_dict({"timesteps_per_hour": tph})
        assert tc.timesteps_per_hour == tph

    def test_zone_target_temperatures(self) -> None:
        tc = TaskConfig.from_dict(
            {
                "zone_target_temperatures": {
                    "Core Zone": {"occupied_c": 23.0, "unoccupied_c": 19.0},
                },
            }
        )
        target = tc.target_for_zone("Core Zone")
        assert target.occupied_c == 23.0
        target_default = tc.target_for_zone("Unknown Zone")
        assert target_default.occupied_c == 21.0

    def test_zone_targets_non_mapping_raises(self) -> None:
        with pytest.raises(TypeError, match="zone_target_temperatures"):
            TaskConfig.from_dict({"zone_target_temperatures": "bad"})

    def test_expected_steps_delegates(self) -> None:
        tc = TaskConfig.from_dict({"run_period": "summer", "timesteps_per_hour": 6})
        assert tc.expected_steps() == 92 * 24 * 6

    def test_random_schedule_mode_accepted(self) -> None:
        tc = TaskConfig.from_dict({"target_temperature_mode": "random_schedule"})
        assert tc.target_temperature_mode == "random_schedule"
        assert tc.random_schedule_config is not None

    def test_random_schedule_config_from_dict(self) -> None:
        tc = TaskConfig.from_dict(
            {
                "target_temperature_mode": "random_schedule",
                "random_schedule": {"building_type": "OfficeSmall", "seed": 42},
            }
        )
        assert tc.random_schedule_config is not None
        assert tc.random_schedule_config.building_type == "OfficeSmall"
        assert tc.random_schedule_config.seed == 42

    def test_random_schedule_requires_mapping(self) -> None:
        with pytest.raises(TypeError, match="random_schedule"):
            TaskConfig.from_dict(
                {
                    "target_temperature_mode": "random_schedule",
                    "random_schedule": "not a mapping",
                }
            )

    def test_random_schedule_config_defaults(self) -> None:
        rs = RandomScheduleConfig()
        assert rs.building_type is None
        assert rs.seed == 0


@pytest.mark.quick
class TestZoneTargetTemperatureConfigExtended:
    def test_from_dict_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="unoccupied_policy"):
            ZoneTargetTemperatureConfig.from_dict(
                {
                    "occupied_c": 21.0,
                    "unoccupied_policy": "spring_only",
                },
                fallback_temperature_c=21.0,
            )

    def test_from_dict_seasonal_unknown_season_key_raises(self) -> None:
        with pytest.raises(ValueError, match="seasonal_unoccupied_c"):
            ZoneTargetTemperatureConfig.from_dict(
                {
                    "occupied_c": 21.0,
                    "unoccupied_policy": "seasonal",
                    "seasonal_unoccupied_c": {
                        "winter": 18.0,
                        "shoulder": 21.0,
                        "summer": 26.0,
                        "monsoon": 24.0,
                    },
                },
                fallback_temperature_c=21.0,
            )


@pytest.mark.quick
class TestRewardConfigs:
    def test_normalized_deadband_reward_config(self) -> None:
        cfg = NormalizedDeadbandRewardConfig(energy_weight=0.01, dT=1.0)
        assert cfg.energy_weight == 0.01
        assert cfg.dT == 1.0
        assert not cfg.is_filled

    def test_normalized_deadband_reward_config_filled(self) -> None:
        cfg = NormalizedDeadbandRewardConfig(
            energy_weight=1.0, dT=1.0, tau_T=0.4, tau_E=0.7
        )
        assert cfg.is_filled


@pytest.mark.quick
class TestRewardConfigFromDict:
    def test_normalized_deadband(self) -> None:
        cfg = reward_config_from_dict(
            {
                "reward_type": "NormalizedDeadbandRewardConfig",
                "energy_weight": 1.0,
                "dT": 1.0,
            }
        )
        assert isinstance(cfg, NormalizedDeadbandRewardConfig)
        assert cfg.energy_weight == 1.0

    def test_missing_reward_type_raises(self) -> None:
        with pytest.raises(ValueError, match="reward_type is required"):
            reward_config_from_dict({})

    def test_unknown_reward_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reward_type"):
            reward_config_from_dict({"reward_type": "FancyReward"})
