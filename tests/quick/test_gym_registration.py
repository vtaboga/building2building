"""Tests for building2building.envs.registration — Gymnasium env registration."""
# This file pins the public API contract.
# Changes here = breaking API changes; requires a CHANGELOG.md entry.
# Marker applied automatically by conftest.py (api_contract glob).


from __future__ import annotations

import gymnasium as gym
import pytest

from building2building.data.download import ALL_BUILDING_TYPES
from building2building.envs.registration import register_all


@pytest.mark.quick
class TestGymRegistration:
    def test_all_building_types_registered(self) -> None:
        import building2building  # noqa: F401

        assert "b2b/OfficeSmall-v0" in gym.envs.registration.registry

        register_all()
        registry_keys = set(gym.envs.registration.registry.keys())
        for bt in ALL_BUILDING_TYPES:
            env_id = f"b2b/{bt}-v0"
            assert env_id in registry_keys, f"{env_id} not registered"

    def test_idempotent_registration(self) -> None:
        register_all()
        register_all()
        count = sum(1 for k in gym.envs.registration.registry if k.startswith("b2b/"))
        assert count == len(ALL_BUILDING_TYPES)

    def test_env_spec_entry_point(self) -> None:
        register_all()
        spec = gym.spec("b2b/OfficeSmall-v0")
        assert (
            spec.entry_point
            == "building2building.envs.registration:make_registered_env"
        )
