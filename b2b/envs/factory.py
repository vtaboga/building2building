from __future__ import annotations

import gymnasium as gym
from pathlib import Path

from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig, reward_to_dict
from b2b.datasets import access as dataset_access
from b2b.simulator import create_simulator


def _build_query_for_selection(selection: DatasetSelectionConfig, building_id: int) -> dict[str, object]:
    if selection.dataset == "single_zone_houses":
        return {
            "idf_filename": f"IDFsAndSchedules/{building_id}/in.idf",
            "schedule_filename": f"IDFsAndSchedules/{building_id}/in.schedules.csv",
        }
    if selection.dataset == "multizones_reference_buildings":
        out: dict[str, object] = {"building_id": int(building_id)}
        if selection.building_type is not None:
            out["building_type"] = selection.building_type
        return out
    raise ValueError(f"Unsupported dataset: {selection.dataset!r}")


def make_env_from_config(config: EnvBuildConfig, eplus_output_dir: str | Path) -> gym.Env:
    out_dir = Path(eplus_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = dataset_access.select_building_ids(config.dataset_selection)
    if len(selected_ids) < 1:
        raise RuntimeError("dataset selection produced zero buildings")
    selected_building_id = int(selected_ids[0])
    bldg_query = _build_query_for_selection(config.dataset_selection, selected_building_id)
    search_cfg = {
        "bldg": {"bldg": bldg_query},
        "reward": reward_to_dict(config.reward),
        "task": {
            "run_period": config.task.run_period.name,
            "target_temperature_mode": config.task.target_temperature_mode,
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
        max_steps = config.task.run_period.expected_steps()
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(max_steps))
