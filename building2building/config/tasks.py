"""Named task presets from the Building2Building paper (Section 4).

Each preset fully specifies the reward function, target temperature mode,
and temperature setpoints for a reproducible benchmark task.

Five presets are currently defined:

* ``task1`` / ``task2`` / ``task4``: constant 21 °C target; vary in
  reward family (deadband vs barrier) and energy weight.
* ``task3``: occupancy-based target with a **seasonal** unoccupied
  policy — winter 18 °C, shoulder 21 °C, summer 26 °C.  This avoids
  the counter-productive "drive unoccupied zones to 18 °C in July"
  behaviour of the original paper preset.
* ``task5``: random daily schedule — each simulated day, a fresh
  arrival time, departure time, occupied setpoint, and unoccupied
  setpoint are sampled from a per-building-type distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from building2building.types import (
    DEFAULT_SEASONAL_UNOCCUPIED_C,
    BarrierRewardConfig,
    DeadbandRewardConfig,
    RewardConfig,
    SeasonName,
    TargetTemperatureMode,
    UnoccupiedPolicy,
)


@dataclass(frozen=True)
class TaskPreset:
    """A named combination of reward and task parameters.

    Attributes:
        reward: Reward function configuration.
        target_temperature_mode: How target temperature varies
            (``"constant"``, ``"occupancy"``-based, or
            ``"random_schedule"``).
        target_temperature_occupied: Target when zone is occupied
            (°C).  Used as the *fallback* occupied setpoint for
            ``"random_schedule"``.
        target_temperature_unoccupied: Target when zone is unoccupied
            (°C).  Used when ``unoccupied_policy == "fixed"``; acts
            as fallback otherwise.
        unoccupied_policy: ``"fixed"`` (default) keeps
            :attr:`target_temperature_unoccupied` constant;
            ``"seasonal"`` dispatches to :attr:`seasonal_unoccupied_c`
            based on the simulation month.
        seasonal_unoccupied_c: Per-season unoccupied setpoints (°C)
            used when ``unoccupied_policy == "seasonal"``.  ``None``
            means "fall back to :data:`DEFAULT_SEASONAL_UNOCCUPIED_C`"
            at the resolution site.
    """

    reward: RewardConfig
    target_temperature_mode: TargetTemperatureMode
    target_temperature_occupied: float
    target_temperature_unoccupied: float
    unoccupied_policy: UnoccupiedPolicy = "fixed"
    seasonal_unoccupied_c: dict[SeasonName, float] | None = field(default=None)


TASK_PRESETS: dict[str, TaskPreset] = {
    "task1": TaskPreset(
        reward=DeadbandRewardConfig(energy_weight=0.01, dT=1.0),
        target_temperature_mode="constant",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=21.0,
    ),
    "task2": TaskPreset(
        reward=DeadbandRewardConfig(energy_weight=0.10, dT=1.0),
        target_temperature_mode="constant",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=21.0,
    ),
    "task3": TaskPreset(
        reward=DeadbandRewardConfig(energy_weight=0.01, dT=1.0),
        target_temperature_mode="occupancy",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=18.0,
        unoccupied_policy="seasonal",
        seasonal_unoccupied_c=dict(DEFAULT_SEASONAL_UNOCCUPIED_C),
    ),
    "task4": TaskPreset(
        reward=BarrierRewardConfig(
            energy_weight=0.01, dT=1.0, violation_penalty=10.0
        ),
        target_temperature_mode="constant",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=21.0,
    ),
    "task5": TaskPreset(
        reward=DeadbandRewardConfig(energy_weight=0.01, dT=1.0),
        target_temperature_mode="random_schedule",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=18.0,
    )
}


def resolve_task_preset(task: str) -> TaskPreset:
    """Look up a named task preset.

    Args:
        task: One of ``"task1"``, ``"task2"``, ``"task3"``,
            ``"task4"``, or ``"task5"``.

    Returns:
        The corresponding :class:`TaskPreset`.

    Raises:
        KeyError: If *task* is not a recognised preset name.
    """
    if task not in TASK_PRESETS:
        raise KeyError(
            f"Unknown task preset {task!r}. "
            f"Available: {sorted(TASK_PRESETS.keys())}"
        )
    return TASK_PRESETS[task]
