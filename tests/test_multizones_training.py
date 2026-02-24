"""Tests for the multizones training script and custom policy loading.

These tests mock EnergyPlus and dataset I/O so they run fast without
external dependencies.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from b2b.baselines.multizones_trainer import (
    SPLIT_DATA_DIR,
    _resolve_building_id,
    load_split_ids,
)
from b2b.baselines.policies import make_policy_from_config
from b2b.sources.multizones_reference_buildings import BuildingType

ALL_TYPES: list[BuildingType] = [
    "Warehouse",
    "HotelSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]


# ── split loading ────────────────────────────────────────────────


class TestLoadSplitIds:
    @pytest.mark.parametrize("building_type", ALL_TYPES)
    def test_train_split_has_900_ids(self, building_type: BuildingType) -> None:
        ids = load_split_ids(building_type, "train")
        assert len(ids) == 900
        assert all(isinstance(i, int) for i in ids)

    @pytest.mark.parametrize("building_type", ALL_TYPES)
    def test_test_split_has_100_ids(self, building_type: BuildingType) -> None:
        ids = load_split_ids(building_type, "test")
        assert len(ids) == 100
        assert all(isinstance(i, int) for i in ids)

    @pytest.mark.parametrize("building_type", ALL_TYPES)
    def test_train_and_test_are_disjoint(self, building_type: BuildingType) -> None:
        train = set(load_split_ids(building_type, "train"))
        test = set(load_split_ids(building_type, "test"))
        assert train.isdisjoint(test), f"Overlap in {building_type}: {train & test}"

    def test_missing_split_file_raises(self, tmp_path: Path) -> None:
        with patch.object(
            __import__("b2b.baselines.multizones_trainer", fromlist=["SPLIT_DATA_DIR"]),
            "SPLIT_DATA_DIR",
            tmp_path,
        ):
            # Re-import to pick up the patched constant
            from b2b.baselines import multizones_trainer as mt

            orig = mt.SPLIT_DATA_DIR
            mt.SPLIT_DATA_DIR = tmp_path
            try:
                with pytest.raises(FileNotFoundError):
                    mt.load_split_ids("Warehouse", "train")
            finally:
                mt.SPLIT_DATA_DIR = orig


class TestResolveBuildingId:
    def test_valid_index_returns_id(self) -> None:
        ids = load_split_ids("OfficeSmall", "train")
        assert _resolve_building_id("OfficeSmall", "train", 0) == ids[0]
        assert _resolve_building_id("OfficeSmall", "train", 5) == ids[5]

    def test_negative_index_raises(self) -> None:
        with pytest.raises(IndexError):
            _resolve_building_id("OfficeSmall", "train", -1)

    def test_out_of_range_index_raises(self) -> None:
        with pytest.raises(IndexError):
            _resolve_building_id("OfficeSmall", "train", 9999)


# ── make_policy_from_config ──────────────────────────────────────


class _DummyPolicy:
    """Minimal policy satisfying PolicyLike."""

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        return np.zeros(1), None


class _NoPredictClass:
    pass


class TestMakePolicyCustom:
    def test_custom_loads_class_with_predict(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "custom",
                    "module": "tests.test_multizones_training",
                    "class_name": "_DummyPolicy",
                }
            }
        )
        policy = make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))
        action, state = policy.predict(np.zeros(4))
        assert action is not None

    def test_custom_passes_kwargs(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "custom",
                    "module": "tests.test_multizones_training",
                    "class_name": "_DummyPolicy",
                    "kwargs": {},
                }
            }
        )
        policy = make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))
        assert hasattr(policy, "predict")

    def test_custom_rejects_class_without_predict(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "custom",
                    "module": "tests.test_multizones_training",
                    "class_name": "_NoPredictClass",
                }
            }
        )
        with pytest.raises(TypeError, match="predict"):
            make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))

    def test_custom_missing_module_raises(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "custom",
                    "module": "",
                    "class_name": "Foo",
                }
            }
        )
        with pytest.raises(ValueError, match="module"):
            make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))

    def test_custom_missing_class_name_raises(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "custom",
                    "module": "os",
                    "class_name": "",
                }
            }
        )
        with pytest.raises(ValueError, match="class_name"):
            make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))


class TestMakePolicySb3:
    def test_sb3_missing_algorithm_raises(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "sb3",
                    "algorithm": "",
                    "checkpoint_path": "/tmp/fake.zip",
                }
            }
        )
        with pytest.raises(ValueError, match="algorithm"):
            make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))

    def test_sb3_missing_checkpoint_raises(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "sb3",
                    "algorithm": "ppo",
                    "checkpoint_path": "",
                }
            }
        )
        with pytest.raises(ValueError, match="checkpoint_path"):
            make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))

    def test_sb3_nonexistent_file_raises(self) -> None:
        cfg = OmegaConf.create(
            {
                "policy": {
                    "type": "sb3",
                    "algorithm": "ppo",
                    "checkpoint_path": "/tmp/nonexistent_model_xyz.zip",
                }
            }
        )
        with pytest.raises(FileNotFoundError):
            make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))

    def test_sb3_loads_model_from_checkpoint(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "model.zip"
        ckpt.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_cls = MagicMock()
        mock_cls.load.return_value = mock_model
        mock_module = MagicMock()
        mock_module.PPO = mock_cls

        with patch("importlib.import_module", return_value=mock_module):
            cfg = OmegaConf.create(
                {
                    "policy": {
                        "type": "sb3",
                        "algorithm": "ppo",
                        "checkpoint_path": str(ckpt),
                    }
                }
            )
            policy = make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))
            assert policy is mock_model
            mock_cls.load.assert_called_once_with(str(ckpt))


class TestMakePolicyUnknown:
    def test_unsupported_type_raises(self) -> None:
        cfg = OmegaConf.create({"policy": {"type": "nonexistent_policy"}})
        with pytest.raises(NotImplementedError):
            make_policy_from_config(OmegaConf.to_container(cfg, resolve=True))


# ── make_multizones_env ──────────────────────────────────────────


class TestMakeMultizonesEnv:
    @patch("b2b.baselines.multizones_trainer.make_multizones_env_api")
    def test_creates_env_for_valid_index(
        self,
        mock_make_env_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from b2b.baselines.multizones_trainer import make_multizones_env

        mock_env = MagicMock()
        mock_env.observation_space = MagicMock()
        mock_env.action_space = MagicMock()
        mock_make_env_api.return_value = mock_env

        env = make_multizones_env(
            building_type="OfficeSmall",
            split="train",
            index=0,
            eplus_output_dir=tmp_path,
        )

        mock_make_env_api.assert_called_once()
        assert env is not None

    def test_out_of_range_index_raises(self, tmp_path: Path) -> None:
        from b2b.baselines.multizones_trainer import make_multizones_env

        with pytest.raises(IndexError):
            make_multizones_env(
                building_type="OfficeSmall",
                split="train",
                index=99999,
                eplus_output_dir=tmp_path,
            )


# ── multizones_trainer ───────────────────────────────────────────


class TestMultizonesTrainer:
    @patch("b2b.baselines.multizones_trainer.load_best_model", return_value=None)
    @patch("b2b.baselines.multizones_trainer.build_sb3_model")
    @patch("b2b.baselines.multizones_trainer._make_callbacks")
    @patch("b2b.baselines.multizones_trainer.init_wandb_from_config", return_value=(None, False))
    @patch("b2b.baselines.multizones_trainer._make_envs")
    def test_trainer_runs_full_loop(
        self,
        mock_make_envs: MagicMock,
        mock_wandb: MagicMock,
        mock_make_cbs: MagicMock,
        mock_build_model: MagicMock,
        mock_load_best: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify the trainer wires together env creation, model build, and learn()."""
        mock_train_env = MagicMock()
        mock_eval_env = MagicMock()
        mock_make_envs.return_value = (mock_train_env, mock_eval_env)
        mock_make_cbs.return_value = MagicMock()

        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        cfg = OmegaConf.create(
            {
                "multizones": {
                    "building_type": "OfficeSmall",
                    "split": "train",
                    "index": 0,
                },
                "env": {"normalize_obs": False, "max_steps": 10},
                "reward": {"energy_weight": 0.0},
                "training": {
                    "total_timesteps": 100,
                    "num_train_envs": 1,
                    "eval_freq": 50,
                    "eval_episodes": 1,
                    "cb_gradient_save_freq": 500,
                },
                "policy": {"algorithm": "ppo", "policy_type": "MlpPolicy"},
                "seed": 42,
            }
        )

        from b2b.baselines.multizones_trainer import multizones_trainer

        multizones_trainer(config=cfg, output_dir=tmp_path)

        mock_make_envs.assert_called_once()
        mock_build_model.assert_called_once()
        mock_model.learn.assert_called_once()
        mock_model.save.assert_called_once()
