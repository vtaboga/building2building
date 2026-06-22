"""Quick tests for the seasonal unoccupied setpoint policy (task3 fix)."""

from __future__ import annotations

import pytest

from building2building.config.tasks import TASK_PRESETS
from building2building.simulator.observation_spaces import DynamicTargetTemperature
from building2building.types import ZoneTargetTemperatureConfig


class _StubReader:
    """Callable that returns a fixed value regardless of state."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, _state: object) -> float:
        return self.value


@pytest.mark.quick
class TestZoneTargetTemperatureConfigSeasonal:
    def test_seasonal_requires_all_three_seasons(self) -> None:
        with pytest.raises(ValueError, match="seasonal"):
            ZoneTargetTemperatureConfig(
                occupied_c=21.0,
                unoccupied_c=18.0,
                unoccupied_policy="seasonal",
                seasonal_unoccupied_c={"winter": 18.0, "summer": 26.0},
            )

    def test_seasonal_missing_map_raises(self) -> None:
        with pytest.raises(ValueError, match="seasonal_unoccupied_c is required"):
            ZoneTargetTemperatureConfig(
                occupied_c=21.0,
                unoccupied_c=18.0,
                unoccupied_policy="seasonal",
            )

    def test_unoccupied_for_season_seasonal(self) -> None:
        cfg = ZoneTargetTemperatureConfig(
            occupied_c=21.0,
            unoccupied_c=18.0,
            unoccupied_policy="seasonal",
            seasonal_unoccupied_c={"winter": 17.0, "shoulder": 21.0, "summer": 26.0},
        )
        assert cfg.unoccupied_for_season("winter") == 17.0
        assert cfg.unoccupied_for_season("shoulder") == 21.0
        assert cfg.unoccupied_for_season("summer") == 26.0

    def test_unoccupied_for_season_fixed_ignores_season(self) -> None:
        cfg = ZoneTargetTemperatureConfig(occupied_c=21.0, unoccupied_c=18.0)
        assert cfg.unoccupied_for_season("winter") == 18.0
        assert cfg.unoccupied_for_season("summer") == 18.0

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="unoccupied_policy"):
            ZoneTargetTemperatureConfig(
                occupied_c=21.0,
                unoccupied_c=18.0,
                unoccupied_policy="autumnal",  # type: ignore[arg-type]
            )

    def test_from_dict_seasonal(self) -> None:
        cfg = ZoneTargetTemperatureConfig.from_dict(
            {
                "occupied_c": 21.0,
                "unoccupied_c": 18.0,
                "unoccupied_policy": "seasonal",
                "seasonal_unoccupied_c": {
                    "winter": 18.0,
                    "shoulder": 21.0,
                    "summer": 26.0,
                },
            },
            fallback_temperature_c=21.0,
        )
        assert cfg.unoccupied_policy == "seasonal"
        assert cfg.unoccupied_for_season("summer") == 26.0

    def test_from_dict_seasonal_defaults_filled(self) -> None:
        cfg = ZoneTargetTemperatureConfig.from_dict(
            {"occupied_c": 21.0, "unoccupied_policy": "seasonal"},
            fallback_temperature_c=21.0,
        )
        assert cfg.seasonal_unoccupied_c is not None
        assert set(cfg.seasonal_unoccupied_c.keys()) == {"winter", "shoulder", "summer"}


@pytest.mark.quick
class TestDynamicTargetTemperatureSeasonal:
    def _make(
        self,
        *,
        occupancy: float,
        policy: str,
        month: int,
    ) -> float:
        zone_target = ZoneTargetTemperatureConfig(
            occupied_c=21.0,
            unoccupied_c=18.0,
            unoccupied_policy=policy,  # type: ignore[arg-type]
            seasonal_unoccupied_c=(
                {"winter": 17.0, "shoulder": 21.0, "summer": 26.0}
                if policy == "seasonal"
                else None
            ),
        )
        target = DynamicTargetTemperature(
            occupancy_reader=_StubReader(occupancy),  # type: ignore[arg-type]
            mode="occupancy",
            zone_target=zone_target,
            month_reader=_StubReader(float(month)),
        )
        return target(state=None)  # type: ignore[arg-type]

    def test_occupied_ignores_season(self) -> None:
        assert self._make(occupancy=1.0, policy="seasonal", month=1) == 21.0
        assert self._make(occupancy=5.0, policy="seasonal", month=7) == 21.0

    def test_unoccupied_seasonal_winter(self) -> None:
        assert self._make(occupancy=0.0, policy="seasonal", month=1) == 17.0

    def test_unoccupied_seasonal_shoulder(self) -> None:
        assert self._make(occupancy=0.0, policy="seasonal", month=5) == 21.0

    def test_unoccupied_seasonal_summer(self) -> None:
        assert self._make(occupancy=0.0, policy="seasonal", month=7) == 26.0

    def test_fixed_policy_ignores_month(self) -> None:
        assert self._make(occupancy=0.0, policy="fixed", month=7) == 18.0

    def test_seasonal_without_month_reader_raises(self) -> None:
        zone_target = ZoneTargetTemperatureConfig(
            occupied_c=21.0,
            unoccupied_c=18.0,
            unoccupied_policy="seasonal",
            seasonal_unoccupied_c={"winter": 17.0, "shoulder": 21.0, "summer": 26.0},
        )
        target = DynamicTargetTemperature(
            occupancy_reader=_StubReader(0.0),  # type: ignore[arg-type]
            mode="occupancy",
            zone_target=zone_target,
            month_reader=None,
        )
        with pytest.raises(RuntimeError, match="month_reader"):
            target(state=None)  # type: ignore[arg-type]


@pytest.mark.quick
class TestTask3IsSeasonal:
    def test_task3_uses_seasonal_policy(self) -> None:
        preset = TASK_PRESETS["task_occ_emed"]
        assert preset.unoccupied_policy == "seasonal"
        assert preset.seasonal_unoccupied_c is not None
        assert preset.seasonal_unoccupied_c["winter"] == 18.0
        assert preset.seasonal_unoccupied_c["summer"] == 26.0
