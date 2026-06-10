"""Gymnasium environment registration for Building2Building.

Registers one ``gym.make``-compatible environment per building type
under the ``b2b/`` namespace.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from building2building.data.download import ALL_BUILDING_TYPES, BuildingType


def make_registered_env(
    building_type: BuildingType,
    split: str = "train",
    index: int = 0,
    task: str = "task_const_e0",
    run_period: str = "full_year",
    timesteps_per_hour: int = 12,
    eplus_output_dir: str | None = None,
    **kwargs: Any,
) -> gym.Env:
    """Entry point used by ``gym.make()`` for registered B2B environments.

    This function is referenced via ``entry_point`` in the registration
    calls below.  It delegates to :func:`building2building.api.make_env`,
    the high-level functional API.
    """
    from building2building.api import make_env

    return make_env(
        building_type=building_type,
        split=split,
        index=index,
        task=task,
        run_period=run_period,
        timesteps_per_hour=timesteps_per_hour,
        eplus_output_dir=eplus_output_dir,
        **kwargs,
    )


def register_all() -> None:
    """Register a ``b2b/<BuildingType>-v0`` environment for each type."""
    for bt in ALL_BUILDING_TYPES:
        env_id = f"b2b/{bt}-v0"
        if env_id not in gym.envs.registration.registry:
            gym.register(
                id=env_id,
                entry_point="building2building.envs.registration:make_registered_env",
                kwargs={"building_type": bt},
            )
