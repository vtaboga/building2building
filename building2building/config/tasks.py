"""Named task presets from the Building2Building paper (Section 4).

Each preset fully specifies the reward function, target temperature mode,
and temperature setpoints for a reproducible benchmark task.
"""

from __future__ import annotations

from dataclasses import dataclass

from building2building.types import (
    BarrierRewardConfig,
    DeadbandRewardConfig,
    RewardConfig,
    TargetTemperatureMode,
)


@dataclass(frozen=True)
class TaskPreset:
    """A named combination of reward and task parameters.

    Attributes:
        reward: Reward function configuration.
        target_temperature_mode: How target temperature varies
            (``"constant"`` or ``"occupancy"``-based).
        target_temperature_occupied: Target when zone is occupied (°C).
        target_temperature_unoccupied: Target when zone is unoccupied (°C).
    """

    reward: RewardConfig
    target_temperature_mode: TargetTemperatureMode
    target_temperature_occupied: float
    target_temperature_unoccupied: float


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
    ),
    "task4": TaskPreset(
        reward=BarrierRewardConfig(
            energy_weight=0.01, dT=1.0, violation_penalty=10.0
        ),
        target_temperature_mode="constant",
        target_temperature_occupied=21.0,
        target_temperature_unoccupied=21.0,
    ),
}


def resolve_task_preset(task: str) -> TaskPreset:
    """Look up a named task preset.

    Args:
        task: One of ``"task1"``, ``"task2"``, ``"task3"``, ``"task4"``.

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
