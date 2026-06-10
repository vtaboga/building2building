"""Quick tests for the random daily schedule module."""

from __future__ import annotations

import pytest

from building2building.simulator.schedules import (
    BUILDING_TYPE_DEFAULTS,
    BuildingTypeScheduleDistribution,
    DailySchedule,
    RandomDailyScheduleGenerator,
    TruncatedNormal,
    distribution_for_building_type,
    month_to_season,
)


@pytest.mark.quick
class TestDailySchedule:
    def test_arrival_before_departure(self) -> None:
        sched = DailySchedule(
            arrival_h=8.0, departure_h=17.0, occupied_c=21.0, unoccupied_c=18.0
        )
        assert sched.arrival_h < sched.departure_h

    def test_arrival_after_departure_raises(self) -> None:
        with pytest.raises(ValueError, match="arrival_h"):
            DailySchedule(
                arrival_h=18.0, departure_h=8.0, occupied_c=21.0, unoccupied_c=18.0
            )

    def test_equal_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="arrival_h"):
            DailySchedule(
                arrival_h=10.0, departure_h=10.0, occupied_c=21.0, unoccupied_c=18.0
            )

    def test_is_occupied_inside_window(self) -> None:
        sched = DailySchedule(8.0, 17.0, 21.0, 18.0)
        assert sched.is_occupied(12.0)
        assert sched.is_occupied(8.0)  # inclusive lower
        assert not sched.is_occupied(17.0)  # exclusive upper
        assert not sched.is_occupied(7.9)
        assert not sched.is_occupied(23.0)


@pytest.mark.quick
class TestMonthToSeason:
    @pytest.mark.parametrize("m", [12, 1, 2])
    def test_winter(self, m: int) -> None:
        assert month_to_season(m) == "winter"

    @pytest.mark.parametrize("m", [6, 7, 8])
    def test_summer(self, m: int) -> None:
        assert month_to_season(m) == "summer"

    @pytest.mark.parametrize("m", [3, 4, 5, 9, 10, 11])
    def test_shoulder(self, m: int) -> None:
        assert month_to_season(m) == "shoulder"

    def test_invalid_month_raises(self) -> None:
        with pytest.raises(ValueError, match="month"):
            month_to_season(0)
        with pytest.raises(ValueError, match="month"):
            month_to_season(13)


@pytest.mark.quick
class TestBuildingTypeDefaults:
    def test_all_paper_types_have_defaults(self) -> None:
        expected = {
            "OfficeSmall",
            "OfficeMedium",
            "RestaurantFastFood",
            "RetailStandalone",
            "Warehouse",
            "SingleFamilyHouse",
        }
        assert expected.issubset(BUILDING_TYPE_DEFAULTS.keys())

    def test_unknown_type_falls_back(self) -> None:
        dist = distribution_for_building_type("CustomArchetype")
        assert isinstance(dist, BuildingTypeScheduleDistribution)

    def test_none_type_falls_back(self) -> None:
        dist = distribution_for_building_type(None)
        assert isinstance(dist, BuildingTypeScheduleDistribution)


@pytest.mark.quick
class TestRandomDailyScheduleGenerator:
    def _gen(self, seed: int = 0) -> RandomDailyScheduleGenerator:
        return RandomDailyScheduleGenerator(
            distribution=BUILDING_TYPE_DEFAULTS["OfficeSmall"],
            base_seed=seed,
        )

    def test_determinism_same_seed(self) -> None:
        g1 = self._gen(seed=42)
        g2 = self._gen(seed=42)
        s1 = g1.schedule_for(year=2023, day_of_year=10, season="winter")
        s2 = g2.schedule_for(year=2023, day_of_year=10, season="winter")
        assert s1 == s2

    def test_different_seeds_differ(self) -> None:
        s1 = self._gen(seed=1).schedule_for(year=2023, day_of_year=10, season="winter")
        s2 = self._gen(seed=2).schedule_for(year=2023, day_of_year=10, season="winter")
        assert s1 != s2

    def test_varies_across_days(self) -> None:
        g = self._gen()
        schedules = [
            g.schedule_for(year=2023, day_of_year=d, season="shoulder")
            for d in range(1, 20)
        ]
        distinct_arrivals = {round(s.arrival_h, 4) for s in schedules}
        assert len(distinct_arrivals) > 1

    def test_varies_across_building_types(self) -> None:
        office = RandomDailyScheduleGenerator(
            BUILDING_TYPE_DEFAULTS["OfficeSmall"], base_seed=0
        )
        warehouse = RandomDailyScheduleGenerator(
            BUILDING_TYPE_DEFAULTS["Warehouse"], base_seed=0
        )
        office_arrivals = [
            office.schedule_for(year=2023, day_of_year=d, season="shoulder").arrival_h
            for d in range(1, 30)
        ]
        warehouse_arrivals = [
            warehouse.schedule_for(
                year=2023, day_of_year=d, season="shoulder"
            ).arrival_h
            for d in range(1, 30)
        ]
        # Warehouse arrives on average earlier than office.
        assert sum(warehouse_arrivals) / len(warehouse_arrivals) < sum(
            office_arrivals
        ) / len(office_arrivals)

    def test_setpoint_bounds(self) -> None:
        g = self._gen()
        for d in range(1, 366):
            s = g.schedule_for(year=2023, day_of_year=d, season="shoulder")
            dist = BUILDING_TYPE_DEFAULTS["OfficeSmall"]
            assert dist.occupied_c_low <= s.occupied_c <= dist.occupied_c_high
            assert dist.unoccupied_c_min <= s.unoccupied_c <= dist.unoccupied_c_max
            assert dist.arrival.low <= s.arrival_h <= dist.arrival.high
            assert dist.departure.low <= s.departure_h <= dist.departure.high
            assert s.arrival_h < s.departure_h

    def test_winter_bias_unoccupied_below_occupied(self) -> None:
        """In winter, ``unoccupied_c`` should never exceed ``occupied_c``
        because the season bias clips the offset window to ``<= 0``."""
        g = self._gen()
        for d in range(1, 60):
            s = g.schedule_for(year=2023, day_of_year=d, season="winter")
            assert s.unoccupied_c <= s.occupied_c + 1e-6

    def test_summer_bias_unoccupied_above_occupied(self) -> None:
        g = self._gen()
        for d in range(1, 60):
            s = g.schedule_for(year=2023, day_of_year=d, season="summer")
            assert s.unoccupied_c >= s.occupied_c - 1e-6


@pytest.mark.quick
class TestTruncatedNormal:
    def test_sample_within_bounds(self) -> None:
        import numpy as np

        dist = TruncatedNormal(mean=8.0, std=0.5, low=6.0, high=10.0)
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = dist.sample(rng)
            assert 6.0 <= x <= 10.0

    def test_invalid_std_raises(self) -> None:
        with pytest.raises(ValueError, match="std"):
            TruncatedNormal(mean=0.0, std=0.0, low=-1.0, high=1.0)

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="low"):
            TruncatedNormal(mean=0.0, std=1.0, low=2.0, high=1.0)
