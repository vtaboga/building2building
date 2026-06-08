"""Unit tests for ``NormalizedDeadbandReward`` and its dispatch wiring.

These tests stub the observation dict directly (no EnergyPlus) and the
dispatch tests build a minimal :class:`BuildingConfig`-like object that
:func:`building2building.simulator.create_simulator` would normally
receive — but only as far as exercising the dispatch helpers; we don't
construct an EnergyPlus simulation.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from building2building.simulator import _maybe_warn_normalized_deadband
from building2building.simulator.rewards import NormalizedDeadbandReward
from building2building.types import (
    DEFAULT_SEASONAL_UNOCCUPIED_C,
    NormalizedDeadbandRewardConfig,
    RunPeriodConfig,
    TaskConfig,
    ZoneTargetTemperatureConfig,
)


def _make_task_config(
    *,
    target_temperature_mode: str = "occupancy",
) -> TaskConfig:
    """Build a minimal TaskConfig in the calibration regime by default."""
    if target_temperature_mode == "occupancy":
        zone = ZoneTargetTemperatureConfig(
            occupied_c=21.0,
            unoccupied_c=18.0,
            unoccupied_policy="seasonal",
            seasonal_unoccupied_c=dict(DEFAULT_SEASONAL_UNOCCUPIED_C),
        )
    else:
        zone = ZoneTargetTemperatureConfig(
            occupied_c=21.0, unoccupied_c=21.0, unoccupied_policy="fixed"
        )
    return TaskConfig(
        run_period=RunPeriodConfig.from_name("full_year"),
        target_temperature_mode=target_temperature_mode,  # type: ignore[arg-type]
        default_zone_target_temperature=zone,
        zone_target_temperatures={},
    )


def _make_obs(
    *,
    zone: str = "z1",
    temp_c: float = 22.0,
    target_c: float = 21.0,
    electricity_wh_m2: float = 1.5,
    natural_gas_wh_m2: float = 0.5,
) -> dict[str, Any]:
    return {
        "temperature": {zone: temp_c},
        "target_temperature": {zone: target_c},
        "energy": {
            "electricity": electricity_wh_m2,
            "natural_gas": natural_gas_wh_m2,
        },
    }


@pytest.mark.quick
class TestNormalizedDeadbandRewardConfig:
    def test_unfilled_preset_state_is_valid(self) -> None:
        cfg = NormalizedDeadbandRewardConfig(energy_weight=1.0, dT=1.0)
        assert not cfg.is_filled
        assert cfg.tau_T is None
        assert cfg.tau_E is None

    def test_filled_state_is_valid(self) -> None:
        cfg = NormalizedDeadbandRewardConfig(
            energy_weight=1.0, dT=1.0, tau_T=0.4, tau_E=0.7
        )
        assert cfg.is_filled

    def test_mixed_unfilled_state_raises(self) -> None:
        with pytest.raises(ValueError, match="must be set together"):
            NormalizedDeadbandRewardConfig(
                energy_weight=1.0, dT=1.0, tau_T=0.4, tau_E=None
            )

    def test_zero_or_negative_tau_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            NormalizedDeadbandRewardConfig(
                energy_weight=1.0, dT=1.0, tau_T=0.0, tau_E=0.7
            )

    def test_filled_method_replaces_taus(self) -> None:
        cfg = NormalizedDeadbandRewardConfig(energy_weight=1.0, dT=1.0)
        filled = cfg.filled(0.4, 0.7)
        assert filled.is_filled
        assert filled.tau_T == 0.4
        assert filled.tau_E == 0.7
        # original is unchanged (frozen dataclass + replace())
        assert not cfg.is_filled


@pytest.mark.quick
class TestNormalizedDeadbandRewardFormula:
    def test_reward_is_squared_temp_error_plus_scaled_energy(self) -> None:
        # |dev|=1.5: temp_error = 1.5^2 = 2.25; energy = 1.5+0.5 = 2.0
        # reward = -(2.25/1.0 + 0.5 * 2.0/1.0) = -3.25
        task_cfg = _make_task_config(target_temperature_mode="constant")
        obs = _make_obs(temp_c=22.5, target_c=21.0)
        reward_fn = NormalizedDeadbandReward(
            controlled_zones=["z1"],
            energy_weight=0.5,
            dT=1.0,
            tau_T=1.0,
            tau_E=1.0,
            task_config=task_cfg,
        )
        assert reward_fn(obs) == pytest.approx(-3.25)

    def test_tau_two_halves_temp_contribution(self) -> None:
        # |dev|=2.5: temp_error = 2.5^2 = 6.25; energy_weight=0
        # tau_T=2.0: reward = -(6.25/2.0) = -3.125
        task_cfg = _make_task_config(target_temperature_mode="constant")
        obs = _make_obs(
            temp_c=23.5, target_c=21.0, electricity_wh_m2=0.0, natural_gas_wh_m2=0.0
        )
        reward_fn = NormalizedDeadbandReward(
            controlled_zones=["z1"],
            energy_weight=0.0,
            dT=1.0,
            tau_T=2.0,
            tau_E=1.0,
            task_config=task_cfg,
        )
        assert reward_fn(obs) == pytest.approx(-3.125)


@pytest.mark.quick
class TestCalibrationWarnings:
    """Tests for ``_maybe_warn_normalized_deadband``.

    The helper deduplicates per process via a module-level ``set``, so
    we ensure each test case gets a fresh ``(bt, bid, mode, dT)`` tuple
    by mixing distinct ``building_id`` strings.
    """

    def test_calibration_regime_does_not_warn(self) -> None:
        task_cfg = _make_task_config(target_temperature_mode="occupancy")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _maybe_warn_normalized_deadband(
                task_config=task_cfg,
                dT=1.0,
                building_type="OfficeMedium",
                building_id="OfficeMedium-quiet-1",
                tau_T=0.4,
                tau_E=0.7,
            )

    def test_non_unit_dT_warns(self) -> None:
        task_cfg = _make_task_config(target_temperature_mode="occupancy")
        with pytest.warns(RuntimeWarning, match="dT"):
            _maybe_warn_normalized_deadband(
                task_config=task_cfg,
                dT=2.0,
                building_type="OfficeMedium",
                building_id="OfficeMedium-warn-dt-1",
                tau_T=0.4,
                tau_E=0.7,
            )

    def test_non_occupancy_mode_warns(self) -> None:
        task_cfg = _make_task_config(target_temperature_mode="constant")
        with pytest.warns(RuntimeWarning, match="target_temperature_mode"):
            _maybe_warn_normalized_deadband(
                task_config=task_cfg,
                dT=1.0,
                building_type="OfficeMedium",
                building_id="OfficeMedium-warn-mode-1",
                tau_T=0.4,
                tau_E=0.7,
            )

    def test_random_schedule_mode_warns(self) -> None:
        task_cfg = _make_task_config(target_temperature_mode="random_schedule")
        with pytest.warns(RuntimeWarning, match="target_temperature_mode"):
            _maybe_warn_normalized_deadband(
                task_config=task_cfg,
                dT=1.0,
                building_type="OfficeMedium",
                building_id="OfficeMedium-warn-mode-2",
                tau_T=0.4,
                tau_E=0.7,
            )

    def test_repeated_warning_is_suppressed(self) -> None:
        task_cfg = _make_task_config(target_temperature_mode="constant")
        # First call: warns. Subsequent identical calls: silent.
        with pytest.warns(RuntimeWarning):
            _maybe_warn_normalized_deadband(
                task_config=task_cfg,
                dT=1.0,
                building_type="OfficeMedium",
                building_id="OfficeMedium-warn-dedup-1",
                tau_T=0.4,
                tau_E=0.7,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _maybe_warn_normalized_deadband(
                task_config=task_cfg,
                dT=1.0,
                building_type="OfficeMedium",
                building_id="OfficeMedium-warn-dedup-1",
                tau_T=0.4,
                tau_E=0.7,
            )
