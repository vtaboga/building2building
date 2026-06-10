"""Morphology must account for every observation and action slot.

``build_morphology`` maps each flat observation / action slot onto a graph
node.  Any slot left unmapped is silently dropped (observations never reach a
node; actions are zeroed by ``join_actions``).  Both ``unassigned_obs_indices``
and ``unassigned_action_indices`` must therefore be empty -- a non-empty one
means the NodeTypes in ``building2building.morphology`` are out of sync with
the slots the pipeline emits (e.g. an actuator type with no node to land on).

One case per building class: the parametrization is derived from the
``BuildingType`` literal, so a new building type added to the dataset is
automatically required to have full morphology coverage (and a fixture).
"""

from __future__ import annotations

import typing
from typing import Any

import pytest

import building2building.api as api_mod
from building2building.data.registry import BuildingType
from building2building.types import RewardConfig

_BUILDING_TYPES: tuple[str, ...] = typing.get_args(BuildingType)


@pytest.mark.quick
@pytest.mark.parametrize(
    "minimal_building_dir",
    _BUILDING_TYPES,
    indirect=True,
)
def test_morphology_has_no_unassigned_slots(
    monkeypatch: pytest.MonkeyPatch,
    fixture_registry: Any,
    minimal_building_dir: Any,
) -> None:
    # ``minimal_building_dir`` resolves the building-type string to its
    # committed minimal fixture; ``make_env`` uses the same string.
    building_type = fixture_registry.entry["building_type"]
    monkeypatch.setattr(
        "building2building.data.registry.get_registry", lambda: fixture_registry
    )
    env = api_mod.make_env(
        building_type,
        task="task_occ_emed",
        reward=RewardConfig(energy_weight=1.0, dT=1.0, tau_T=1.0, tau_E=1.0),
        run_period="winter",
        target_temperature_mode="constant",
        max_episode_steps=5,
        rescale_action=False,
    )
    try:
        metadata = env.unwrapped.metadata
        morphology = metadata["morphology"]
        obs_names = metadata["observation_names"]
        act_names = metadata["action_names"]

        unassigned_obs = [obs_names[i] for i in morphology.unassigned_obs_indices]
        unassigned_act = [act_names[i] for i in morphology.unassigned_action_indices]

        assert morphology.unassigned_obs_indices == (), (
            f"{building_type}: observation slots not mapped to any morphology "
            f"node: {unassigned_obs}"
        )
        assert morphology.unassigned_action_indices == (), (
            f"{building_type}: action slots not mapped to any morphology node "
            f"(join_actions would zero them): {unassigned_act}"
        )
    finally:
        env.close()
