"""Factory for creating Gymnasium environments from build configs.

Bridges the configuration layer (:mod:`building2building.config.models`) with the
dataset access and EnergyPlus simulator layers.
"""

from __future__ import annotations

import gymnasium as gym
from pathlib import Path

from building2building.config.models import DatasetSelectionConfig, EnvBuildConfig, reward_to_dict
from building2building.datasets import access as dataset_access
from building2building.simulator import create_simulator


def _build_bldg_section(selection: DatasetSelectionConfig, building_id: int) -> dict[str, object]:
    """Build the ``"bldg"`` section of the search config dict.

    For single_zone_houses the query filters go under a ``"query"`` sub-key.
    For multizones_reference_buildings the filters are top-level keys.
    """
    if selection.dataset == "single_zone_houses":
        return {
            "query": {
                "idf_filename": f"IDFsAndSchedules/{building_id}/in.idf",
                "schedule_filename": f"IDFsAndSchedules/{building_id}/in.schedules.csv",
            },
        }
    if selection.dataset == "multizones_reference_buildings":
        out: dict[str, object] = {"building_id": int(building_id)}
        if selection.building_type is not None:
            out["building_type"] = selection.building_type
        return out
    raise ValueError(f"Unsupported dataset: {selection.dataset!r}")


def make_env_from_config(config: EnvBuildConfig, eplus_output_dir: str | Path) -> gym.Env:
    """Construct a time-limited Gymnasium environment from an ``EnvBuildConfig``.

    The function selects a building from the configured dataset, builds a
    search config for the dataset access layer, creates an EnergyPlus
    simulator, and wraps it in a :class:`gymnasium.wrappers.TimeLimit`.

    Args:
        config: Fully specified environment build configuration.
        eplus_output_dir: Directory where EnergyPlus writes simulation
            artefacts. Created if it does not exist.

    Returns:
        A :class:`gymnasium.wrappers.TimeLimit`-wrapped EnergyPlus
        environment.

    Raises:
        RuntimeError: If the dataset selection yields zero buildings.
        ValueError: If the configured dataset is not supported.
    """
    out_dir = Path(eplus_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = dataset_access.select_building_ids(config.dataset_selection)
    if len(selected_ids) < 1:
        raise RuntimeError("dataset selection produced zero buildings")
    selected_building_id = int(selected_ids[0])
    bldg_section = _build_bldg_section(config.dataset_selection, selected_building_id)
    search_cfg = {
        "bldg": bldg_section,
        "reward": reward_to_dict(config.reward),
        "task": {
            "run_period": config.task.run_period.name,
            "target_temperature_mode": config.task.target_temperature_mode,
            "timesteps_per_hour": config.task.timesteps_per_hour,
            "default_zone_target_temperature": {
                "occupied_c": config.task.default_zone_target_temperature.occupied_c,
                "unoccupied_c": config.task.default_zone_target_temperature.unoccupied_c,
            },
            "zone_target_temperatures": {
                zone: {
                    "occupied_c": tgt.occupied_c,
                    "unoccupied_c": tgt.unoccupied_c,
                }
                for zone, tgt in config.task.zone_target_temperatures.items()
            },
        },
        "expose_heating_only_zones": config.expose_heating_only_zones,
    }

    if config.dataset_selection.dataset == "single_zone_houses":
        built = dataset_access.search_config(
            dataset="single_zone_houses",
            config=search_cfg,
            eplus_output_dir=out_dir,
        )
    else:
        built = dataset_access.search_config(
            dataset="multizones_reference_buildings",
            building_type=config.dataset_selection.building_type,
            config=search_cfg,
            eplus_output_dir=out_dir,
        )
    env = create_simulator(built)
    max_steps = config.env_max_steps
    if max_steps is None:
        max_steps = config.task.expected_steps()
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(max_steps))
