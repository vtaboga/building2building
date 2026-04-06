"""Public API for creating Building2Building Gymnasium environments.

Provides convenience functions that wrap the lower-level config / factory
machinery so that users can create single-zone or multi-zone EnergyPlus
environments with minimal boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from typing import Literal

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

    equipment_data = json.loads(equipment_path.read_text())
    from building2building.simulator import create_simulator

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


def make_single_zone_env(
    *,
    split: str,
    split_index: int,
    eplus_output_dir: str | Path,
    reward: dict,
    task: dict | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    """Create a single-zone house environment by split and index.

    This is a convenience wrapper that builds the full
    :class:`~building2building.config.models.EnvBuildConfig` from scalar arguments
    targeting the ``single_zone_houses`` dataset.

    Args:
        split: Dataset split to use (``"train"`` or ``"test"``).
        split_index: Zero-based index into the chosen split.
        eplus_output_dir: Directory for EnergyPlus output files.
        reward: Reward configuration dictionary.  Must contain a
            ``"reward_type"`` key.  See
            :func:`~building2building.types.reward_config_from_dict` for accepted keys.
        task: Optional task configuration dictionary. See
            :meth:`~building2building.types.TaskConfig.from_dict` for accepted keys.
        max_steps: Maximum episode length. Defaults to the number of
            simulation steps implied by the run period.

    Returns:
        A Gymnasium environment for the selected single-zone house.
    """
    task_cfg = TaskConfig.from_dict(task or {})
    reward_cfg = reward_config_from_dict(reward)
    return make_env_from_config(
        EnvBuildConfig(
            dataset_selection=DatasetSelectionConfig(
                dataset="single_zone_houses",
                split=split,  # type: ignore[arg-type]
                mode="split_index",
                split_index=int(split_index),
            ),
            task=task_cfg,
            reward=reward_cfg,
            env_max_steps=max_steps,
        ),
        eplus_output_dir=eplus_output_dir,
    )


def make_multizones_env(
    *,
    building_type: str,
    split: str,
    split_index: int,
    eplus_output_dir: str | Path,
    reward: dict,
    task: dict | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    """Create a multi-zone reference-building environment.

    This is a convenience wrapper that builds the full
    :class:`~building2building.config.models.EnvBuildConfig` from scalar arguments
    targeting the ``multizones_reference_buildings`` dataset.

    Args:
        building_type: Reference building type (e.g. ``"OfficeSmall"``).
        split: Dataset split to use (``"train"`` or ``"test"``).
        split_index: Zero-based index into the chosen split.
        eplus_output_dir: Directory for EnergyPlus output files.
        reward: Reward configuration dictionary.  Must contain a
            ``"reward_type"`` key.  See
            :func:`~building2building.types.reward_config_from_dict` for accepted keys.
        task: Optional task configuration dictionary. See
            :meth:`~building2building.types.TaskConfig.from_dict` for accepted keys.
        max_steps: Maximum episode length. Defaults to the number of
            simulation steps implied by the run period.

    Returns:
        A Gymnasium environment for the selected multi-zone building.
    """
    task_cfg = TaskConfig.from_dict(task or {})
    reward_cfg = reward_config_from_dict(reward)
    return make_env_from_config(
        EnvBuildConfig(
            dataset_selection=DatasetSelectionConfig(
                dataset="multizones_reference_buildings",
                building_type=building_type,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
                mode="split_index",
                split_index=int(split_index),
            ),
            task=task_cfg,
            reward=reward_cfg,
            env_max_steps=max_steps,
        ),
        eplus_output_dir=eplus_output_dir,
    )


def _to_plain_dict(config: object) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    try:
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(config, resolve=True)
        if isinstance(raw, dict):
            return dict(raw)
    except Exception:
        pass
    return {}


def _infer_dataset_selection(cfg: dict[str, Any]) -> DatasetSelectionConfig:
    bldg = cfg.get("bldg", {})
    if not isinstance(bldg, dict):
        bldg = {}
    dataset = str(bldg.get("dataset", "single_zone_houses"))
    building_type = bldg.get("building_type")
    split = str(bldg.get("split", "train"))
    index = int(bldg.get("index", 0))
    query = bldg.get("query", {})
    if not isinstance(query, dict):
        query = {}
    climate_zone = bldg.get("climate_zone")

    if (
        climate_zone is not None
        and dataset == "multizones_reference_buildings"
        and building_type is not None
    ):
        from building2building.sources.multizones_reference_buildings import (
            find_building_id_for_climate_zone,
        )

        bid = find_building_id_for_climate_zone(
            building_type=building_type,
            split=split,  # type: ignore[arg-type]
            climate_zone=int(climate_zone),
            index=index,
        )
        return DatasetSelectionConfig(
            dataset=dataset,
            building_type=str(building_type),
            split=split,  # type: ignore[arg-type]
            mode="building_id",
            building_id=bid,
        )

    if query:
        return DatasetSelectionConfig(
            dataset=dataset,
            building_type=str(building_type) if building_type else None,
            split=split,  # type: ignore[arg-type]
            mode="metadata_query",
            metadata_query=dict(query),
            sample_size=1,
        )
    return DatasetSelectionConfig(
        dataset=dataset,
        building_type=str(building_type) if building_type else None,
        split=split,  # type: ignore[arg-type]
        mode="split_index",
        split_index=index,
    )


def make_env_from_hydra_config(
    config: object, eplus_output_dir: str | Path
) -> gym.Env:
    """Create a Gymnasium environment from a Hydra/OmegaConf config object.

    This is a compatibility bridge for training scripts that use Hydra
    configuration.  It converts the untyped config dict into a typed
    :class:`~building2building.config.models.EnvBuildConfig` and delegates
    to :func:`make_env`.

    Args:
        config: A Hydra/OmegaConf config or plain dict with ``bldg``,
            ``reward``, ``task``, and optionally ``env`` sections.
        eplus_output_dir: Directory for EnergyPlus output files.

    Returns:
        A configured Gymnasium environment backed by EnergyPlus.
    """
    import json
    import logging
    import traceback
    import uuid

    logger = logging.getLogger(__name__)
    out_dir = Path(eplus_output_dir) / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cfg = _to_plain_dict(config)
        task_section = (
            cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}
        )
        reward_raw = cfg.get("reward")
        if not isinstance(reward_raw, dict) or not reward_raw:
            raise ValueError(
                "The 'reward' section is required in the environment config.  "
                "Pass a dict with at least a 'reward_type' key."
            )
        reward_section: dict[str, Any] = reward_raw
        env_section = (
            cfg.get("env", {}) if isinstance(cfg.get("env"), dict) else {}
        )
        task = TaskConfig.from_dict(task_section)
        reward = reward_config_from_dict(reward_section)
        build = EnvBuildConfig(
            dataset_selection=_infer_dataset_selection(cfg),
            task=task,
            reward=reward,
            env_max_steps=(
                int(env_section["max_steps"])
                if env_section.get("max_steps") is not None
                else None
            ),
            expose_heating_only_zones=bool(
                env_section.get("expose_heating_only_zones", True)
            ),
        )
        return make_env(build, eplus_output_dir=out_dir)
    except Exception as e:
        err_path = Path(out_dir) / "env_creation_error.json"
        record: dict[str, Any] = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            with err_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception:
            logger.exception(
                "Failed to write env creation error to %s", err_path
            )
        raise


__all__ = [
    "list_building_types",
    "list_buildings",
    "make_env",
    "make_env_from_hydra_config",
    "make_multizones_env",
    "make_single_zone_env",
    "new_make_env",
    "parse_benchmark_config",
]
