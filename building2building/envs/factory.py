"""Factory for creating Gymnasium environments from build configs.

Bridges the configuration layer (:mod:`building2building.config.models`) with the
unified dataset registry and EnergyPlus simulator.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
from cattrs import structure

from building2building.config.models import EnvBuildConfig, reward_to_dict
from building2building.data.registry import get_registry
from building2building.pipeline.actuators import AnyEquipment
from building2building.simulator import create_simulator
from building2building.types import BuildingConfig, TaskConfig


def make_env_from_config(
    config: EnvBuildConfig, eplus_output_dir: str | Path
) -> gym.Env:
    """Construct a time-limited Gymnasium environment from an ``EnvBuildConfig``.

    Resolves the building via the unified :class:`BuildingRegistry`, then
    creates an EnergyPlus simulator and wraps it in a
    :class:`gymnasium.wrappers.TimeLimit`.

    Args:
        config: Fully specified environment build configuration.
        eplus_output_dir: Directory where EnergyPlus writes simulation
            artefacts. Created if it does not exist.

    Returns:
        A :class:`gymnasium.wrappers.TimeLimit`-wrapped EnergyPlus
        environment.
    """
    out_dir = Path(eplus_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = config.dataset_selection
    registry = get_registry()

    if sel.mode == "building_id" and sel.building_id is not None:
        info = registry.get_building_by_id(sel.building_type, sel.building_id)
    else:
        split = sel.split or "train"
        info = registry.get_building_by_index(sel.building_type, split, sel.split_index)

    epjson_path = info.building_dir / "building.epjson"
    equipment_path = info.building_dir / "equipment.json"
    weather_path = info.building_dir / info.weather_file
    equipment_data = structure(
        json.loads(equipment_path.read_text()), list[AnyEquipment]
    )

    building_config = BuildingConfig(
        path_to_building=epjson_path,
        path_to_weather=weather_path,
        reward_config=config.reward,
        eplus_output_dir=out_dir,
        warmup_phases=info.warmup_phases,
        area=info.net_conditioned_area_m2,
        hvac_equipment=equipment_data,
        task_config=config.task,
        expose_heating_only_zones=config.expose_heating_only_zones,
    )

    env = create_simulator(building_config)
    env.metadata["building_info"] = info
    max_steps = config.env_max_steps
    if max_steps is None:
        max_steps = config.task.expected_steps()
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(max_steps))
