"""Random daily occupancy/setpoint schedules for ``task_rand_*`` presets.

Defines per-building-type distributions for arrival and departure times
as well as occupied/unoccupied setpoint ranges, and provides a
deterministic generator that samples a fresh :class:`DailySchedule`
each simulated day.

This module is deliberately side-effect-free and EnergyPlus-agnostic so
that it can be unit-tested quickly without launching a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from building2building.types import SeasonName

# ---------------------------------------------------------------------------
# Season resolver
# ---------------------------------------------------------------------------

WINTER_MONTHS: frozenset[int] = frozenset({12, 1, 2})
SUMMER_MONTHS: frozenset[int] = frozenset({6, 7, 8})


def month_to_season(month: int) -> SeasonName:
    """Map an EnergyPlus month (1–12) to a season bucket.

    ``"winter"`` covers DEC–FEB, ``"summer"`` covers JUN–AUG, and
    everything else is ``"shoulder"``.

    Raises:
        ValueError: If *month* is outside ``[1, 12]``.
    """
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be in [1, 12], got {month}")
    m = int(month)
    if m in WINTER_MONTHS:
        return "winter"
    if m in SUMMER_MONTHS:
        return "summer"
    return "shoulder"


# ---------------------------------------------------------------------------
# Daily schedule + distribution dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailySchedule:
    """A sampled schedule for a single simulated day.

    Attributes:
        arrival_h: Hour of day when the zone becomes occupied
            (0.0–24.0).
        departure_h: Hour of day when the zone becomes unoccupied
            (must be strictly greater than :attr:`arrival_h`).
        occupied_c: Target temperature during the occupied window
            (°C).
        unoccupied_c: Target temperature outside the occupied window
            (°C).
    """

    arrival_h: float
    departure_h: float
    occupied_c: float
    unoccupied_c: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.arrival_h < 24.0:
            raise ValueError(f"arrival_h must be in [0, 24), got {self.arrival_h}")
        if not 0.0 < self.departure_h <= 24.0:
            raise ValueError(f"departure_h must be in (0, 24], got {self.departure_h}")
        if self.arrival_h >= self.departure_h:
            raise ValueError(
                "arrival_h must be strictly less than departure_h, "
                f"got arrival_h={self.arrival_h} departure_h={self.departure_h}"
            )

    def is_occupied(self, hour_of_day: float) -> bool:
        """Return whether *hour_of_day* falls in the occupied window."""
        return self.arrival_h <= hour_of_day < self.departure_h


@dataclass(frozen=True)
class TruncatedNormal:
    """Truncated-normal distribution with hard lower/upper clips.

    Samples are drawn from ``N(mean, std)`` and clipped to the
    ``[low, high]`` interval via rejection (up to 10 tries, then
    deterministic clip).
    """

    mean: float
    std: float
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.std <= 0.0:
            raise ValueError(f"std must be > 0, got {self.std}")
        if self.low >= self.high:
            raise ValueError(f"low must be < high, got low={self.low} high={self.high}")

    def sample(self, rng: np.random.Generator) -> float:
        for _ in range(10):
            x = float(rng.normal(self.mean, self.std))
            if self.low <= x <= self.high:
                return x
        return float(np.clip(rng.normal(self.mean, self.std), self.low, self.high))


@dataclass(frozen=True)
class BuildingTypeScheduleDistribution:
    """Per-building-type distribution of daily schedules.

    The occupied setpoint is drawn uniformly from
    ``[occupied_c_low, occupied_c_high]``.  The unoccupied setpoint is
    drawn uniformly from
    ``[occupied + unoccupied_offset_low, occupied + unoccupied_offset_high]``
    and clipped to ``[unoccupied_c_min, unoccupied_c_max]``.  The sign
    of the offset is biased by season: in winter we allow drift down,
    in summer we allow drift up, and in the shoulder we keep a narrow
    window around the occupied value.
    """

    arrival: TruncatedNormal
    departure: TruncatedNormal
    occupied_c_low: float
    occupied_c_high: float
    unoccupied_offset_low: float
    unoccupied_offset_high: float
    unoccupied_c_min: float = 14.0
    unoccupied_c_max: float = 30.0

    def __post_init__(self) -> None:
        if self.occupied_c_low >= self.occupied_c_high:
            raise ValueError(
                "occupied_c_low must be < occupied_c_high, got "
                f"{self.occupied_c_low} vs {self.occupied_c_high}"
            )
        if self.unoccupied_offset_low > self.unoccupied_offset_high:
            raise ValueError("unoccupied_offset_low must be <= unoccupied_offset_high")
        if self.unoccupied_c_min >= self.unoccupied_c_max:
            raise ValueError("unoccupied_c_min must be < unoccupied_c_max")

    def sample(self, rng: np.random.Generator, season: SeasonName) -> DailySchedule:
        """Draw a full :class:`DailySchedule` for a given *season*."""
        arrival_h = self.arrival.sample(rng)
        departure_h = self.departure.sample(rng)
        if departure_h <= arrival_h:
            # Enforce positive occupied window with a safety margin.
            departure_h = min(24.0, arrival_h + 1.0)
        occupied_c = float(rng.uniform(self.occupied_c_low, self.occupied_c_high))
        season_bias = _season_offset_bias(season)
        low, high = _apply_season_bias(
            self.unoccupied_offset_low,
            self.unoccupied_offset_high,
            season_bias,
        )
        unoccupied_c_raw = float(occupied_c + rng.uniform(low, high))
        unoccupied_c = float(
            np.clip(unoccupied_c_raw, self.unoccupied_c_min, self.unoccupied_c_max)
        )
        return DailySchedule(
            arrival_h=arrival_h,
            departure_h=departure_h,
            occupied_c=occupied_c,
            unoccupied_c=unoccupied_c,
        )


_SeasonBias = Literal["negative", "neutral", "positive"]


def _season_offset_bias(season: SeasonName) -> _SeasonBias:
    if season == "winter":
        return "negative"
    if season == "summer":
        return "positive"
    return "neutral"


def _apply_season_bias(
    low: float, high: float, bias: _SeasonBias
) -> tuple[float, float]:
    """Clamp the offset window according to the season bias.

    In ``"winter"`` we force the unoccupied target to drift **down**
    (offset ≤ 0), in ``"summer"`` we force it **up** (offset ≥ 0), and
    in the shoulder we keep the symmetric window supplied by the
    distribution.
    """
    if bias == "negative":
        return (min(low, 0.0), min(high, 0.0) if high > 0 else high)
    if bias == "positive":
        return (max(low, 0.0) if low < 0 else low, max(high, 0.0))
    return (low, high)


# ---------------------------------------------------------------------------
# Per-building-type defaults (matching short keys used across the repo)
# ---------------------------------------------------------------------------

_OFFICE = BuildingTypeScheduleDistribution(
    arrival=TruncatedNormal(mean=8.0, std=0.5, low=6.5, high=10.0),
    departure=TruncatedNormal(mean=17.0, std=0.5, low=15.0, high=20.0),
    occupied_c_low=20.0,
    occupied_c_high=23.0,
    unoccupied_offset_low=-3.0,
    unoccupied_offset_high=3.0,
)

_RESTAURANT = BuildingTypeScheduleDistribution(
    arrival=TruncatedNormal(mean=9.0, std=0.5, low=7.0, high=11.0),
    departure=TruncatedNormal(mean=23.0, std=0.5, low=20.0, high=24.0),
    occupied_c_low=20.0,
    occupied_c_high=23.0,
    unoccupied_offset_low=-3.0,
    unoccupied_offset_high=3.0,
)

_RETAIL = BuildingTypeScheduleDistribution(
    arrival=TruncatedNormal(mean=9.0, std=0.5, low=7.0, high=11.0),
    departure=TruncatedNormal(mean=21.0, std=0.5, low=18.0, high=23.0),
    occupied_c_low=20.0,
    occupied_c_high=23.0,
    unoccupied_offset_low=-3.0,
    unoccupied_offset_high=3.0,
)

_WAREHOUSE = BuildingTypeScheduleDistribution(
    arrival=TruncatedNormal(mean=6.5, std=0.5, low=5.0, high=9.0),
    departure=TruncatedNormal(mean=18.0, std=0.5, low=15.0, high=21.0),
    occupied_c_low=18.0,
    occupied_c_high=22.0,
    unoccupied_offset_low=-4.0,
    unoccupied_offset_high=4.0,
)

_HOUSE = BuildingTypeScheduleDistribution(
    arrival=TruncatedNormal(mean=7.0, std=1.0, low=5.0, high=10.0),
    departure=TruncatedNormal(mean=22.0, std=1.0, low=19.0, high=24.0),
    occupied_c_low=20.0,
    occupied_c_high=23.0,
    unoccupied_offset_low=-3.0,
    unoccupied_offset_high=3.0,
)


BUILDING_TYPE_DEFAULTS: dict[str, BuildingTypeScheduleDistribution] = {
    "OfficeSmall": _OFFICE,
    "OfficeMedium": _OFFICE,
    "RestaurantFastFood": _RESTAURANT,
    "RetailStandalone": _RETAIL,
    "Warehouse": _WAREHOUSE,
    "SingleFamilyHouse": _HOUSE,
}


_GENERIC_FALLBACK = _OFFICE


def distribution_for_building_type(
    building_type: str | None,
) -> BuildingTypeScheduleDistribution:
    """Return the default distribution for *building_type*.

    Falls back to the office distribution when *building_type* is
    ``None`` or unknown.  Unknown types emit no warning because the
    benchmark generator may produce custom archetypes.
    """
    if building_type is None:
        return _GENERIC_FALLBACK
    return BUILDING_TYPE_DEFAULTS.get(building_type, _GENERIC_FALLBACK)


# ---------------------------------------------------------------------------
# Deterministic per-day generator
# ---------------------------------------------------------------------------


def _day_seed(base_seed: int, year: int, day_of_year: int) -> int:
    """Produce a deterministic 63-bit seed for a given calendar day."""
    mixed = (
        (int(base_seed) & 0xFFFF_FFFF) * 1_000_000
        + (int(year) & 0xFFFF) * 1_000
        + (int(day_of_year) & 0x3FF)
    )
    return int(mixed) & 0x7FFF_FFFF_FFFF_FFFF


@dataclass
class RandomDailyScheduleGenerator:
    """Deterministic generator that yields a fresh schedule per day.

    Attributes:
        distribution: Per-building-type distribution used for sampling.
        base_seed: Episode-level seed.  Combined with ``year`` and
            ``day_of_year`` via :func:`_day_seed` so that two calls
            with the same ``(base_seed, year, day_of_year)`` always
            produce identical schedules.
    """

    distribution: BuildingTypeScheduleDistribution
    base_seed: int = 0

    def schedule_for(
        self, *, year: int, day_of_year: int, season: SeasonName
    ) -> DailySchedule:
        rng = np.random.default_rng(_day_seed(self.base_seed, year, day_of_year))
        return self.distribution.sample(rng, season)


__all__ = [
    "BUILDING_TYPE_DEFAULTS",
    "BuildingTypeScheduleDistribution",
    "DailySchedule",
    "RandomDailyScheduleGenerator",
    "TruncatedNormal",
    "distribution_for_building_type",
    "month_to_season",
]
