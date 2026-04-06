"""Public API for creating Building2Building Gymnasium environments.

Provides convenience functions that wrap the lower-level config / factory
machinery so that users can create single-zone or multi-zone EnergyPlus
environments with minimal boilerplate.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from b2b.config.models import DatasetSelectionConfig, EnvBuildConfig, parse_benchmark_config
from b2b.envs import make_env_from_config
from b2b.types import TaskConfig, reward_config_from_dict


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
    :class:`~b2b.config.models.EnvBuildConfig` from scalar arguments
    targeting the ``single_zone_houses`` dataset.

    Args:
        split: Dataset split to use (``"train"`` or ``"test"``).
        split_index: Zero-based index into the chosen split.
        eplus_output_dir: Directory for EnergyPlus output files.
        reward: Reward configuration dictionary.  Must contain a
            ``"reward_type"`` key.  See
            :func:`~b2b.types.reward_config_from_dict` for accepted keys.
        task: Optional task configuration dictionary. See
            :meth:`~b2b.types.TaskConfig.from_dict` for accepted keys.
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
    :class:`~b2b.config.models.EnvBuildConfig` from scalar arguments
    targeting the ``multizones_reference_buildings`` dataset.

    Args:
        building_type: Reference building type (e.g. ``"OfficeSmall"``).
        split: Dataset split to use (``"train"`` or ``"test"``).
        split_index: Zero-based index into the chosen split.
        eplus_output_dir: Directory for EnergyPlus output files.
        reward: Reward configuration dictionary.  Must contain a
            ``"reward_type"`` key.  See
            :func:`~b2b.types.reward_config_from_dict` for accepted keys.
        task: Optional task configuration dictionary. See
            :meth:`~b2b.types.TaskConfig.from_dict` for accepted keys.
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


__all__ = [
    "make_env",
    "make_multizones_env",
    "make_single_zone_env",
    "parse_benchmark_config",
]
