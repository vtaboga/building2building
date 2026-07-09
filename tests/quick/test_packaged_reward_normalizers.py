"""Invariants of the committed ``building2building/data/reward_normalizers.yaml``.

The packaged YAML is a build artifact of
``baselines.compute_reactive_reward_normalizers``; these tests pin the
contract a regeneration must preserve:

* one ``constants`` section per supported run period (``winter``,
  ``summer``, ``full_year``) — a missing section makes every env build
  for that period raise;
* comfort is unnormalized: ``tau_T == 1.0`` (and ``tau_T_iqr == 0.0``)
  in every bucket, so ``temp_penalty`` stays in raw degC^2;
* ``tau_E`` is a strictly positive energy normalizer and the defensive
  floor is dormant on the shipped data.
"""

from __future__ import annotations

import pytest

from building2building.data.reward_normalizers import (
    clear_reward_normalizers_cache,
    load_reward_normalizers,
)

RUN_PERIODS = ("winter", "summer", "full_year")


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_reward_normalizers_cache()


@pytest.mark.quick
@pytest.mark.parametrize("run_period", RUN_PERIODS)
class TestPackagedRewardNormalizers:
    def test_section_present_and_nonempty(self, run_period: str) -> None:
        table = load_reward_normalizers(run_period=run_period)
        assert table.constants, f"no constants for run_period={run_period!r}"

    def test_comfort_is_unnormalized(self, run_period: str) -> None:
        table = load_reward_normalizers(run_period=run_period)
        for building_type, by_cz in table.constants.items():
            for cz_key, bucket in by_cz.items():
                assert bucket.tau_T == 1.0, (
                    f"{run_period}/{building_type}/{cz_key}: tau_T={bucket.tau_T!r}"
                    " (comfort must be unnormalized, tau_T == 1)"
                )
                assert bucket.tau_T_iqr == 0.0, (
                    f"{run_period}/{building_type}/{cz_key}:"
                    f" tau_T_iqr={bucket.tau_T_iqr!r}"
                )

    def test_energy_normalizer_positive_and_floor_dormant(
        self, run_period: str
    ) -> None:
        table = load_reward_normalizers(run_period=run_period)
        for building_type, by_cz in table.constants.items():
            for cz_key, bucket in by_cz.items():
                key = f"{run_period}/{building_type}/{cz_key}"
                assert bucket.tau_E > 0.0, f"{key}: tau_E={bucket.tau_E!r}"
                assert not bucket.floor_applied_T, f"{key}: comfort floor fired"
                assert not bucket.floor_applied_E, f"{key}: energy floor fired"
