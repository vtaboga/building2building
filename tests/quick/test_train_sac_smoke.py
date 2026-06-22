"""Smoke tests for the SAC training harness.

These tests verify that:
1. ``build_sac`` constructs a SAC instance on a stub env without raising.
2. ``PAPER_SAC_DEFAULTS`` has the expected keys and value types.
3. The SAC module imports correctly (no circular dependencies).

All tests are quick (no EnergyPlus, no training steps) and run on the
login node before any SLURM submission.
"""

from __future__ import annotations

import pytest


@pytest.mark.quick
class TestBuildSac:
    def test_build_sac_constructs_without_error(self) -> None:
        """build_sac must return a SAC instance on a minimal Box env."""
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv

        from baselines.utils.training import build_sac

        env = gym.make("Pendulum-v1")
        vec_env = DummyVecEnv([lambda: env])
        try:
            model = build_sac(vec_env, seed=0, verbose=0)
            assert isinstance(model, SAC)
        finally:
            vec_env.close()

    def test_build_sac_accepts_overrides(self) -> None:
        """build_sac must accept hyperparameter overrides."""
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv

        from baselines.utils.training import build_sac

        env = gym.make("Pendulum-v1")
        vec_env = DummyVecEnv([lambda: env])
        try:
            model = build_sac(vec_env, seed=42, verbose=0, batch_size=64, gamma=0.95)
            assert isinstance(model, SAC)
            assert model.batch_size == 64
            assert model.gamma == pytest.approx(0.95)
        finally:
            vec_env.close()

    def test_build_sac_activation_fn_string_conversion(self) -> None:
        """build_sac must convert string activation_fn to torch.nn class."""
        import torch
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv

        from baselines.utils.training import build_sac

        env = gym.make("Pendulum-v1")
        vec_env = DummyVecEnv([lambda: env])
        try:
            model = build_sac(
                vec_env,
                seed=0,
                verbose=0,
                policy_kwargs={
                    "net_arch": {"pi": [64], "qf": [64]},
                    "activation_fn": "Tanh",
                },
            )
            assert isinstance(model, SAC)
            assert model.policy.actor.latent_pi[1].__class__ is torch.nn.Tanh
        finally:
            vec_env.close()


@pytest.mark.quick
class TestPaperSacDefaults:
    def test_required_keys_present(self) -> None:
        from baselines.utils.training import PAPER_SAC_DEFAULTS

        required = {
            "learning_rate",
            "buffer_size",
            "learning_starts",
            "batch_size",
            "tau",
            "gamma",
            "train_freq",
            "gradient_steps",
            "ent_coef",
            "target_update_interval",
            "target_entropy",
            "use_sde",
            "sde_sample_freq",
            "policy_kwargs",
        }
        assert required.issubset(set(PAPER_SAC_DEFAULTS.keys()))

    def test_ent_coef_is_auto(self) -> None:
        from baselines.utils.training import PAPER_SAC_DEFAULTS

        assert PAPER_SAC_DEFAULTS["ent_coef"] == "auto"

    def test_buffer_size_within_ram_budget(self) -> None:
        from baselines.utils.training import PAPER_SAC_DEFAULTS

        assert PAPER_SAC_DEFAULTS["buffer_size"] <= 300_000

    def test_policy_kwargs_net_arch(self) -> None:
        import torch
        from baselines.utils.training import PAPER_SAC_DEFAULTS

        pk = PAPER_SAC_DEFAULTS["policy_kwargs"]
        assert "net_arch" in pk
        assert "pi" in pk["net_arch"]
        assert "qf" in pk["net_arch"]
        assert pk["activation_fn"] is torch.nn.ReLU


@pytest.mark.quick
class TestSacModuleImports:
    def test_training_module_imports(self) -> None:
        from baselines.utils.training import PAPER_SAC_DEFAULTS, build_sac

        assert callable(build_sac)
        assert isinstance(PAPER_SAC_DEFAULTS, dict)

    def test_train_sac_module_importable(self) -> None:
        import baselines.train_sac as train_sac_mod

        assert hasattr(train_sac_mod, "train_and_eval")
        assert hasattr(train_sac_mod, "write_results_csv")
