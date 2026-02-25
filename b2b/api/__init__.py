from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig, parse_benchmark_config
from b2b.envs import make_env_from_config
from b2b.types import TaskConfig, reward_config_from_dict


def make_env(config: EnvBuildConfig, eplus_output_dir: str | Path) -> gym.Env:
    return make_env_from_config(config=config, eplus_output_dir=eplus_output_dir)


def make_single_zone_env(
    *,
    split: str,
    split_index: int,
    eplus_output_dir: str | Path,
    task: dict | None = None,
    reward: dict | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    task_cfg = TaskConfig.from_dict(task or {})
    reward_cfg = reward_config_from_dict(reward or {})
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
    task: dict | None = None,
    reward: dict | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    task_cfg = TaskConfig.from_dict(task or {})
    reward_cfg = reward_config_from_dict(reward or {})
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


__all__ = [
    "make_env",
    "make_multizones_env",
    "make_single_zone_env",
    "parse_benchmark_config",
]
