"""Tests for ``building2building.data.reward_normalizers``.

These tests build synthetic YAML inputs in temp directories and exercise
the loader / floor / resolver in isolation, so they pass without the
real ``building2building/data/reward_normalizers.yaml`` being committed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from building2building.data.reward_normalizers import (
    DEFAULT_REWARD_NORMALIZERS_PATH,
    RewardNormalizer,
    RewardNormalizersUnavailableError,
    SUPPORTED_SCHEMA_VERSION,
    _cz_key_for,
    clear_reward_normalizers_cache,
    load_reward_normalizers,
    parse_reward_normalizers,
    resolve_reward_normalizer,
)


def _good_yaml() -> str:
    """Synthetic YAML covering one CZ-bucketed type and one single-bucket type."""
    return textwrap.dedent("""
        schema_version: 1
        source:
          controller: tuned_reactive
          calibration_task: task3
          run_period: full_year
          split: train
          aggregation: median_over_buildings
          git_sha: deadbeef
          b2b_version: '0.1.0'
        floor:
          epsilon_abs: 1.0e-6
          epsilon_rel: 0.05
        constants:
          OfficeMedium:
            cz1: { tau_T: 0.4, tau_E: 0.7, tau_T_iqr: 0.1, tau_E_iqr: 0.2, n_buildings: 24 }
            cz2: { tau_T: 0.5, tau_E: 0.8, tau_T_iqr: 0.1, tau_E_iqr: 0.2, n_buildings: 30 }
          SingleFamilyHouse:
            cz0: { tau_T: 1.9, tau_E: 0.85, tau_T_iqr: 0.6, tau_E_iqr: 0.3, n_buildings: 152 }
        """).strip()


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "reward_normalizers.yaml"
    p.write_text(body + "\n")
    return p


@pytest.mark.quick
class TestParseRewardNormalizers:
    def test_good_yaml_parses(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, _good_yaml())
        table = load_reward_normalizers(path)
        assert table.schema_version == SUPPORTED_SCHEMA_VERSION
        assert table.source.calibration_task == "task3"
        assert table.source.split == "train"
        assert table.source.controller == "tuned_reactive"
        assert "OfficeMedium" in table.constants
        assert "SingleFamilyHouse" in table.constants
        bucket = table.constants["OfficeMedium"]["cz1"]
        assert isinstance(bucket, RewardNormalizer)
        assert bucket.tau_T == pytest.approx(0.4)
        assert bucket.tau_E == pytest.approx(0.7)
        assert bucket.n_buildings == 24

    def test_unsupported_schema_version_raises(self, tmp_path: Path) -> None:
        body = _good_yaml().replace("schema_version: 1", "schema_version: 999")
        path = _write_yaml(tmp_path, body)
        with pytest.raises(ValueError, match="schema_version"):
            load_reward_normalizers(path)

    def test_missing_file_raises_distinct_error(self, tmp_path: Path) -> None:
        with pytest.raises(RewardNormalizersUnavailableError, match="not found"):
            load_reward_normalizers(tmp_path / "does-not-exist.yaml")

    def test_missing_required_source_keys_raises(self, tmp_path: Path) -> None:
        body = textwrap.dedent("""
            schema_version: 1
            source:
              controller: tuned_reactive
            constants:
              OfficeMedium:
                cz1: { tau_T: 0.4, tau_E: 0.7 }
            """).strip()
        path = _write_yaml(tmp_path, body)
        with pytest.raises(ValueError, match="source"):
            load_reward_normalizers(path)


@pytest.mark.quick
class TestRewardNormalizerFloor:
    def test_floor_dormant_on_good_data(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, _good_yaml())
        table = load_reward_normalizers(path)
        for buckets in table.constants.values():
            for bucket in buckets.values():
                assert not bucket.floor_applied_T
                assert not bucket.floor_applied_E

    def test_floor_clips_pathological_bucket(self, tmp_path: Path) -> None:
        body = textwrap.dedent("""
            schema_version: 1
            source:
              controller: tuned_reactive
              calibration_task: task3
              run_period: full_year
              split: train
              aggregation: median_over_buildings
            floor:
              epsilon_abs: 1.0e-6
              epsilon_rel: 0.5
            constants:
              OfficeMedium:
                cz1: { tau_T: 0.4, tau_E: 0.7, tau_T_iqr: 0.1, tau_E_iqr: 0.2, n_buildings: 24 }
                cz2: { tau_T: 0.5, tau_E: 0.8, tau_T_iqr: 0.1, tau_E_iqr: 0.2, n_buildings: 30 }
                cz3: { tau_T: 1.0e-9, tau_E: 1.0e-9, tau_T_iqr: 0.0, tau_E_iqr: 0.0, n_buildings: 1 }
            """).strip()
        path = _write_yaml(tmp_path, body)
        table = load_reward_normalizers(path)
        clipped = table.constants["OfficeMedium"]["cz3"]
        unclipped = table.constants["OfficeMedium"]["cz1"]
        assert clipped.floor_applied_T
        assert clipped.floor_applied_E
        assert clipped.tau_T > 1e-9
        assert clipped.tau_E > 1e-9
        assert not unclipped.floor_applied_T
        assert not unclipped.floor_applied_E


@pytest.mark.quick
class TestResolveRewardNormalizer:
    def test_sfh_returns_single_bucket_regardless_of_id(self) -> None:
        # SFH has no climate-zone column, so we don't need a registry hit.
        assert _cz_key_for("SingleFamilyHouse", "anything-1234") == "cz0"
        assert _cz_key_for("SingleFamilyHouse", "another-id") == "cz0"

    def test_resolve_for_sfh(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, _good_yaml())
        clear_reward_normalizers_cache()
        bucket = resolve_reward_normalizer("SingleFamilyHouse", "ignored-id", path=path)
        assert bucket.tau_T == pytest.approx(1.9)
        assert bucket.tau_E == pytest.approx(0.85)
        assert bucket.n_buildings == 152

    def test_resolve_unknown_building_type_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, _good_yaml())
        clear_reward_normalizers_cache()
        with pytest.raises(KeyError, match="building_type"):
            resolve_reward_normalizer("Warehouse", "Warehouse-001", path=path)


@pytest.mark.quick
class TestDefaultPathSentinel:
    def test_default_path_constant_points_inside_package(self) -> None:
        assert DEFAULT_REWARD_NORMALIZERS_PATH.name == "reward_normalizers.yaml"
        assert DEFAULT_REWARD_NORMALIZERS_PATH.parent.name == "data"
        assert (
            DEFAULT_REWARD_NORMALIZERS_PATH.exists()
        ), "Committed YAML not found; was it accidentally removed?"
