"""Public API for creating Building2Building Gymnasium environments.

Provides convenience functions that wrap the lower-level config / factory
machinery so that users can create EnergyPlus environments with minimal
boilerplate, backed by the unified HuggingFace dataset.
"""

from __future__ import annotations

import json
import logging
import weakref
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from building2building.api.rollout import (
    Controller,
    Trajectory,
    callable_controller,
    rollout,
)
from building2building.config.models import (
    DatasetSelectionConfig,
    EnvBuildConfig,
    parse_benchmark_config,
)
from building2building.config.tasks import TASK_PRESETS, TaskPreset, resolve_task_preset
from building2building.data.climate_zones import (
    TYPES_WITHOUT_CLIMATE_ZONE,
    ClimateZoneUnavailableError,
)
from building2building.data.download import ALL_BUILDING_TYPES, BuildingType
from building2building.envs import make_env_from_config
from building2building.types import (
    NormalizedDeadbandRewardConfig,
    RandomScheduleConfig,
    RewardConfig,
    RunPeriodConfig,
    TaskConfig,
    ZoneTargetTemperatureConfig,
    reward_config_from_dict,
)

logger = logging.getLogger(__name__)


def _resolve_task_config(
    *,
    preset: TaskPreset,
    run_period_cfg: RunPeriodConfig,
    timesteps_per_hour: int,
    target_temperature_mode: str | None,
    random_schedule_seed: int | None,
    building_type: BuildingType,
) -> TaskConfig:
    """Build the effective TaskConfig used for environment creation."""
    effective_mode: str = (
        target_temperature_mode
        if target_temperature_mode is not None
        else preset.target_temperature_mode
    )

    default_zone_target = ZoneTargetTemperatureConfig(
        occupied_c=preset.target_temperature_occupied,
        unoccupied_c=preset.target_temperature_unoccupied,
        unoccupied_policy=preset.unoccupied_policy,
        seasonal_unoccupied_c=(
            dict(preset.seasonal_unoccupied_c)
            if preset.seasonal_unoccupied_c is not None
            else None
        ),
    )

    random_schedule_cfg: RandomScheduleConfig | None = None
    if effective_mode == "random_schedule":
        random_schedule_cfg = RandomScheduleConfig(
            building_type=building_type,
            seed=int(random_schedule_seed) if random_schedule_seed is not None else 0,
        )

    return TaskConfig(
        run_period=run_period_cfg,
        target_temperature_mode=effective_mode,  # type: ignore[arg-type]
        default_zone_target_temperature=default_zone_target,
        timesteps_per_hour=timesteps_per_hour,
        random_schedule_config=random_schedule_cfg,
    )


def _resolve_effective_reward(
    *,
    preset: TaskPreset,
    reward_override: RewardConfig | None,
    building_type: BuildingType,
    building_id: str,
    run_period: str,
    normalizer_path: Path | None,
) -> RewardConfig:
    """Resolve reward config, auto-filling normalized reward constants."""
    effective_reward: RewardConfig = (
        reward_override if reward_override is not None else preset.reward
    )

    if (
        isinstance(effective_reward, NormalizedDeadbandRewardConfig)
        and not effective_reward.is_filled
    ):
        from building2building.data.reward_normalizers import resolve_reward_normalizer

        normalizer = resolve_reward_normalizer(
            building_type,
            building_id,
            run_period=run_period,
            path=normalizer_path,
        )
        return effective_reward.filled(normalizer.tau_T, normalizer.tau_E)

    return effective_reward


def list_building_types() -> list[str]:
    """Return all available building types."""
    return list(ALL_BUILDING_TYPES)


def list_buildings(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"] = "train",
) -> list[str]:
    """Return building IDs for a given type and split.

    Downloads metadata from HuggingFace on first call.
    """
    from building2building.data.registry import get_registry

    return get_registry().list_buildings(building_type, split)


def list_buildings_by_climate_zone(
    building_type: BuildingType,
    climate_zone: int,
    split: Literal["train", "test", "test_small"] = "train",
) -> list[str]:
    """Return building IDs filtered by ASHRAE climate zone.

    Raises:
        ClimateZoneUnavailableError: If ``building_type`` has no ASHRAE
            climate-zone assignment (e.g. ``SingleFamilyHouse``).
    """
    from building2building.data.registry import get_registry

    return get_registry().list_buildings_by_climate_zone(
        building_type, climate_zone, split
    )


def get_climate_zone(
    building_type: BuildingType,
    building_id: str,
) -> int:
    """Return the ASHRAE climate zone of a building.

    Raises:
        ClimateZoneUnavailableError: If ``building_type`` has no ASHRAE
            climate-zone assignment (e.g. ``SingleFamilyHouse``).
        KeyError: If ``building_id`` is not found in the unified metadata.
        ValueError: If the matching row has a null ``climate_zone`` despite
            the type being mappable (indicates a corrupt / stale parquet).
    """
    from building2building.data.registry import get_registry

    if building_type in TYPES_WITHOUT_CLIMATE_ZONE:
        raise ClimateZoneUnavailableError(
            f"{building_type!r} has no ASHRAE climate-zone assignment"
        )
    info = get_registry().get_building_by_id(building_type, building_id)
    if info.climate_zone is None:
        raise ValueError(
            f"climate_zone is null for {building_type}/{building_id}; "
            "your metadata.parquet may be out of date."
        )
    return info.climate_zone


def _patch_epjson_run_period(
    src_epjson: Path,
    dst_epjson: Path,
    run_period_cfg: RunPeriodConfig,
) -> None:
    """Rewrite the ``RunPeriod`` section of an epJSON file on disk.

    The pre-built dataset buildings have ``full_year`` baked in.  When the
    user requests a different run period (``winter``, ``summer``), we must
    patch the dates so that EnergyPlus actually simulates the right season
    instead of relying only on the ``TimeLimit`` wrapper.

    ``Schedule:File`` objects in residential buildings (e.g. SingleFamilyHouse)
    reference external CSV files via relative paths.  When the patched epJSON
    is written to a different directory (``dst_epjson.parent != src_epjson.parent``),
    EnergyPlus cannot find those files and segfaults.  We therefore rewrite
    any relative ``file_name`` entries to absolute paths rooted at
    ``src_epjson.parent`` before writing the patched file.
    """
    src_dir = src_epjson.parent

    with src_epjson.open() as f:
        epjson: dict[str, Any] = json.load(f)

    rp_obj = epjson.setdefault("RunPeriod", {})
    if not isinstance(rp_obj, dict):
        raise TypeError(
            f"Expected epJSON['RunPeriod'] to be a dict, got {type(rp_obj)}"
        )

    if "Run Period 1" in rp_obj:
        rp1 = rp_obj["Run Period 1"]
    elif rp_obj:
        first_key = next(iter(rp_obj))
        rp1 = rp_obj.pop(first_key)
        rp_obj.clear()
        rp_obj["Run Period 1"] = rp1
    else:
        rp1: dict[str, Any] = {
            "apply_weekend_holiday_rule": "No",
            "begin_year": 2023,
            "day_of_week_for_start_day": "Sunday",
            "end_year": 2023,
            "use_weather_file_daylight_saving_period": "No",
            "use_weather_file_holidays_and_special_days": "No",
            "use_weather_file_rain_indicators": "Yes",
            "use_weather_file_snow_indicators": "Yes",
        }
        rp_obj["Run Period 1"] = rp1

    rp1["begin_day_of_month"] = run_period_cfg.begin_day_of_month
    rp1["begin_month"] = run_period_cfg.begin_month
    rp1["end_day_of_month"] = run_period_cfg.end_day_of_month
    rp1["end_month"] = run_period_cfg.end_month

    for sched_obj in epjson.get("Schedule:File", {}).values():
        file_name = sched_obj.get("file_name", "")
        if file_name and not Path(file_name).is_absolute():
            sched_obj["file_name"] = str((src_dir / file_name).resolve())

    with dst_epjson.open("w") as f:
        json.dump(epjson, f, indent=4)


def make_env(
    building_type: BuildingType,
    *,
    split: Literal["train", "test", "test_small"] = "train",
    index: int = 0,
    building_id: str | None = None,
    task: str | TaskPreset = "task_const_e0",
    reward: str | RewardConfig | None = None,
    run_period: str = "full_year",
    normalizer_path: Path | None = None,
    timesteps_per_hour: int = 12,
    target_temperature_mode: str | None = None,
    random_schedule_seed: int | None = None,
    eplus_output_dir: str | Path | None = None,
    max_episode_steps: int | None = None,
    rescale_action: bool = False,
) -> gym.Env:
    """Create a Gymnasium environment using the unified dataset.

    This is the primary user-facing API.  It resolves named task presets,
    downloads pre-processed buildings from HuggingFace, and constructs
    the EnergyPlus simulation environment.

    Args:
        building_type: Building type (e.g. ``"OfficeSmall"``).
        split: Dataset split (``"train"``, ``"test"``, or ``"test_small"``).
        index: Zero-based index into the split.
        building_id: Explicit building ID, overrides *split*/*index*.
        task: Named task preset or a
            :class:`~building2building.config.tasks.TaskPreset` instance.
            Recognised names are the 9 normalized presets
            ``"task_<mode>_<level>"`` with
            ``mode ∈ {const, occ, rand}`` and ``level ∈ {e0, emed, ehigh}``.
            ``(tau_T, tau_E)`` are auto-resolved from
            :file:`building2building/data/reward_normalizers.yaml`
            using the building's ``(building_type, climate_zone)`` bucket.
            Defaults to ``"task_const_e0"`` (constant setpoint,
            comfort-only).
        reward: Override reward.  If ``None``, uses the task default.
        run_period: Simulation run period name (``"full_year"``,
            ``"winter"``, ``"summer"``).
        normalizer_path: Override the default
            :data:`~building2building.data.reward_normalizers.DEFAULT_REWARD_NORMALIZERS_PATH`
            used to resolve ``(tau_T, tau_E)`` for normalized-reward presets.
            When ``None`` (default), the built-in random-policy YAML is used.
        timesteps_per_hour: Number of simulation steps per hour.
        target_temperature_mode: Override the preset's target mode
            (``"constant"``, ``"occupancy"``, or ``"random_schedule"``).
            When ``None`` (default), the mode is taken from the task
            preset, so that e.g. ``task="task_occ_e0"`` automatically
            uses occupancy-based targets.
        random_schedule_seed: Base seed for the per-day schedule
            generator used by ``task_rand_*`` presets.  ``None`` falls
            back to the value on the preset's task config (default ``0``).
        eplus_output_dir: Directory for EnergyPlus output.  If ``None``,
            a temporary directory is used.
        max_episode_steps: Maximum episode length.
        rescale_action: If ``True``, wrap the simulator with
            :class:`gym.wrappers.RescaleAction` so the agent-facing
            action space is ``[-1, 1]`` per actuator.  The wrapper
            maps actions back to engineering units internally.
            Defaults to ``False`` so that non-RL consumers (reactive
            controllers, benchmark harnesses, manual rollouts) are
            unaffected.  RL training code should use
            :func:`building2building.api.rl_wrappers.wrap_env_for_rl`
            or the ``make_rl_env_fn`` helper instead, which set this
            flag and also apply observation normalisation.

    Returns:
        A Gymnasium environment backed by EnergyPlus.
    """
    import tempfile

    from building2building.data.registry import get_registry
    from building2building.types import BuildingConfig

    if isinstance(task, str):
        preset = resolve_task_preset(task)
    else:
        preset = task

    registry = get_registry()
    if building_id is not None:
        info = registry.get_building_by_id(building_type, building_id)
    else:
        info = registry.get_building_by_index(building_type, split, index)

    effective_reward = _resolve_effective_reward(
        preset=preset,
        reward_override=reward,
        building_type=building_type,
        building_id=getattr(info, "building_id", None) or building_id or "",
        run_period=run_period,
        normalizer_path=normalizer_path,
    )

    if eplus_output_dir is None:
        eplus_output_dir = Path(tempfile.mkdtemp(prefix="b2b_eplus_"))
    else:
        eplus_output_dir = Path(eplus_output_dir)
    eplus_output_dir.mkdir(parents=True, exist_ok=True)

    epjson_path = info.building_dir / "building.epjson"
    equipment_path = info.building_dir / "equipment.json"
    weather_path = info.building_dir / info.weather_file

    run_period_cfg = RunPeriodConfig.from_name(run_period)

    # The dataset ships buildings with full_year baked into the epJSON.
    # Patch the RunPeriod dates when the user requests a different period.
    #
    # The patched file is written to a *separate* staging directory, NOT
    # into eplus_output_dir.  eplus_output_dir is the per-episode output
    # directory that close() removes on every reset; if the patched epjson
    # lived there it would be deleted before the first episode starts.  The
    # staging dir is tied to the env's lifetime via weakref.finalize.
    _epjson_staging_dir: Path | None = None
    if run_period_cfg.name != "full_year":
        _epjson_staging_dir = Path(tempfile.mkdtemp(prefix="b2b_epjson_"))
        patched_epjson_path = _epjson_staging_dir / "building.epjson"
        _patch_epjson_run_period(epjson_path, patched_epjson_path, run_period_cfg)
        epjson_path = patched_epjson_path
        logger.debug(
            "Patched epJSON RunPeriod to %s (%d/%d – %d/%d) → %s",
            run_period_cfg.name,
            run_period_cfg.begin_month,
            run_period_cfg.begin_day_of_month,
            run_period_cfg.end_month,
            run_period_cfg.end_day_of_month,
            patched_epjson_path,
        )

    task_cfg = _resolve_task_config(
        preset=preset,
        run_period_cfg=run_period_cfg,
        timesteps_per_hour=timesteps_per_hour,
        target_temperature_mode=target_temperature_mode,
        random_schedule_seed=random_schedule_seed,
        building_type=building_type,
    )

    from cattrs import structure
    from building2building.pipeline.actuators import AnyEquipment
    from building2building.simulator import create_simulator

    equipment_data = structure(
        json.loads(equipment_path.read_text()), list[AnyEquipment]
    )

    building_config = BuildingConfig(
        path_to_building=epjson_path,
        path_to_weather=weather_path,
        reward_config=effective_reward,
        eplus_output_dir=eplus_output_dir,
        warmup_phases=info.warmup_phases,
        area=info.net_conditioned_area_m2,
        hvac_equipment=equipment_data,
        # Stash building identity so the simulator dispatch site can
        # cite the building in calibration-mismatch warnings.
        source_metadata={
            "building_type": building_type,
            "building_id": getattr(info, "building_id", None),
        },
        task_config=task_cfg,
    )

    env = create_simulator(building_config)
    env.metadata["building_info"] = info

    # Register cleanup for the epjson staging dir (only created when
    # run_period != "full_year").  We attach it to the innermost env
    # object (before any wrappers) so it is not affected by wrapper
    # garbage-collection order.
    if _epjson_staging_dir is not None:
        import shutil as _shutil

        weakref.finalize(env, _shutil.rmtree, _epjson_staging_dir, True)

    if rescale_action:
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    steps = max_episode_steps or task_cfg.expected_steps()
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(steps))


__all__ = [
    "list_building_types",
    "list_buildings",
    "list_buildings_by_climate_zone",
    "get_climate_zone",
    "ClimateZoneUnavailableError",
    "TYPES_WITHOUT_CLIMATE_ZONE",
    "Controller",
    "Trajectory",
    "callable_controller",
    "rollout",
    "make_env",
    "make_env_from_config",
    "parse_benchmark_config",
]
