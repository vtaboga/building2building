"""Quick smoke tests for the new public API surface.

Verifies that the top-level building2building package exposes
the expected attributes.  Detailed tests for each subsystem live
in their own test modules.
"""
# This file pins the public API contract.
# Changes here = breaking API changes; requires a CHANGELOG.md entry.
# Marker applied automatically by conftest.py (api_contract glob).


from __future__ import annotations

import pytest


@pytest.mark.quick
class TestTopLevelImports:
    def test_api_functions_exposed(self) -> None:
        import building2building

        assert hasattr(building2building, "list_building_types")
        assert hasattr(building2building, "list_buildings")
        assert hasattr(building2building, "new_make_env")

    def test_scoring_exposed(self) -> None:
        import building2building

        assert hasattr(building2building, "compute_normalized_score")

    def test_benchmarks_exposed(self) -> None:
        import building2building

        assert hasattr(building2building, "benchmarks")

    def test_reward_configs_exposed(self) -> None:
        import building2building

        assert hasattr(building2building, "NormalizedDeadbandRewardConfig")
        assert hasattr(building2building, "RewardConfig")
