"""Public API for creating Building2Building Gymnasium environments.

Provides convenience functions that wrap the lower-level config / factory
machinery so that users can create EnergyPlus environments with minimal
boilerplate, backed by the unified HuggingFace dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from building2building.config.models import DatasetSelectionConfig, EnvBuildConfig, parse_benchmark_config
from building2building.config.tasks import TASK_PRESETS, TaskPreset, resolve_task_preset
from building2building.data.download import ALL_BUILDING_TYPES, BuildingType
from building2building.envs import make_env_from_config
from building2building.types import RewardConfig, TaskConfig, reward_config_from_dict


def list_building_types() -> list[str]:
    """Return all available building types."""
    return list(ALL_BUILDING_TYPES)


def list_buildings(
    building_type: BuildingType,
    split: Literal["train", "test"] = "train",
) -> list[str]:
    """Return building IDs for a given type and split.

    Downloads metadata from HuggingFace on first call.
    """
    from building2building.data.registry import get_registry

    return get_registry().list_buildings(building_type, split)


def new_make_env(
    building_type: BuildingType,
    *,
    split: Literal["train", "test"] = "train",
    index: int = 0,
    building_id: str | None = None,
    task: str | TaskPreset = "task1",
    reward: str | RewardConfig | None = None,
    run_period: str = "full_year",
    timesteps_per_hour: int = 12,
    target_temperature_mode: str = "constant",
    eplus_output_dir: str | Path | None = None,
    max_episode_steps: int | None = None,
) -> gym.Env:
    """Create a Gymnasium environment using the unified dataset.

    This is the primary user-facing API.  It resolves named task presets,
    downloads pre-processed buildings from HuggingFace, and constructs
    the EnergyPlus simulation environment.

    Args:
        building_type: Building type (e.g. ``"OfficeSmall"``).
        split: Dataset split (``"train"`` or ``"test"``).
        index: Zero-based index into the split.
        building_id: Explicit building ID, overrides *split*/*index*.
        task: Named task preset (``"task1"``–``"task4"``) or a
            :class:`~building2building.config.tasks.TaskPreset` instance.
        reward: Override reward.  If ``None``, uses the task default.
        run_period: Simulation run period name (``"full_year"``,
            ``"winter"``, ``"summer"``).
        timesteps_per_hour: Number of simulation steps per hour.
        target_temperature_mode: ``"constant"`` or ``"occupancy"``.
        eplus_output_dir: Directory for EnergyPlus output.  If ``None``,
            a temporary directory is used.
        max_episode_steps: Maximum episode length.

    Returns:
        A Gymnasium environment backed by EnergyPlus.
    """
    import tempfile

    from building2building.data.registry import get_registry
    from building2building.types import (
        BuildingConfig,
        RunPeriodConfig,
        ZoneTargetTemperatureConfig,
    )

    if isinstance(task, str):
        preset = resolve_task_preset(task)
    else:
        preset = task

    effective_reward = reward if reward is not None else preset.reward

    registry = get_registry()
    if building_id is not None:
        info = registry.get_building_by_id(building_type, building_id)
    else:
        info = registry.get_building_by_index(building_type, split, index)

    if eplus_output_dir is None:
        eplus_output_dir = Path(tempfile.mkdtemp(prefix="b2b_eplus_"))
    else:
        eplus_output_dir = Path(eplus_output_dir)
    eplus_output_dir.mkdir(parents=True, exist_ok=True)

    import json

    epjson_path = info.building_dir / "building.epjson"
    equipment_path = info.building_dir / "equipment.json"
    weather_path = info.building_dir / info.weather_file

    run_period_cfg = RunPeriodConfig.from_name(run_period)
    task_cfg = TaskConfig(
        run_period=run_period_cfg,
        target_temperature_mode=target_temperature_mode,  # type: ignore[arg-type]
        default_zone_target_temperature=ZoneTargetTemperatureConfig(
            occupied_c=preset.target_temperature_occupied,
            unoccupied_c=preset.target_temperature_unoccupied,
        ),
        timesteps_per_hour=timesteps_per_hour,
    )

    from cattrs import structure
    from building2building.pipeline.actuators import AnyEquipment
    from building2building.simulator import create_simulator

    equipment_data = structure(json.loads(equipment_path.read_text()), list[AnyEquipment])

    building_config = BuildingConfig(
        path_to_building=epjson_path,
        path_to_weather=weather_path,
        reward_config=effective_reward,
        eplus_output_dir=eplus_output_dir,
        warmup_phases=info.warmup_phases,
        area=info.net_conditioned_area_m2,
        hvac_equipment=equipment_data,
        task_config=task_cfg,
    )

    env = create_simulator(building_config)
    steps = max_episode_steps or task_cfg.expected_steps()
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(steps))


def make_env(config: EnvBuildConfig, eplus_output_dir: str | Path) -> gym.Env:
    """Create a Gymnasium environment from a fully-specified build config.

    Args:
        config: Complete environment build configuration.
        eplus_output_dir: Directory where EnergyPlus will write its
            simulation output files.

    Returns:
        A configured Gymnasium environment backed by EnergyPlus.
    """
    return make_env_from_config(config=config, eplus_output_dir=eplus_output_dir)


__all__ = [
    "list_building_types",
    "list_buildings",
    "make_env",
    "new_make_env",
    "parse_benchmark_config",
]
