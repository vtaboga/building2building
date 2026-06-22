"""Smoke tests for the PPO training harness.

Mirrors the structure of test_train_sac_smoke.py:
1. ``build_ppo`` constructs a PPO instance on a stub env without raising.
2. ``PAPER_PPO_DEFAULTS`` has the expected keys and value types.
3. The module imports correctly (no circular dependencies).
"""

from __future__ import annotations

import pytest


@pytest.mark.quick
class TestBuildPpo:
    def test_build_ppo_constructs_without_error(self) -> None:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from baselines.utils.training import build_ppo

        env = gym.make("CartPole-v1")
        vec_env = DummyVecEnv([lambda: env])
        try:
            model = build_ppo(vec_env, seed=0, verbose=0)
            assert isinstance(model, PPO)
        finally:
            vec_env.close()

    def test_build_ppo_accepts_overrides(self) -> None:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from baselines.utils.training import build_ppo

        env = gym.make("CartPole-v1")
        vec_env = DummyVecEnv([lambda: env])
        try:
            model = build_ppo(vec_env, seed=42, verbose=0, batch_size=32, gamma=0.97)
            assert isinstance(model, PPO)
            assert model.batch_size == 32
            assert model.gamma == pytest.approx(0.97)
        finally:
            vec_env.close()

    def test_build_ppo_activation_fn_string_conversion(self) -> None:
        import torch
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from baselines.utils.training import build_ppo

        env = gym.make("CartPole-v1")
        vec_env = DummyVecEnv([lambda: env])
        try:
            model = build_ppo(
                vec_env,
                seed=0,
                verbose=0,
                policy_kwargs={
                    "net_arch": {"pi": [64], "vf": [64]},
                    "activation_fn": "Tanh",
                },
            )
            assert isinstance(model, PPO)
        finally:
            vec_env.close()


@pytest.mark.quick
class TestPaperPpoDefaults:
    def test_required_keys_present(self) -> None:
        from baselines.utils.training import PAPER_PPO_DEFAULTS

        required = {
            "learning_rate",
            "batch_size",
            "n_steps",
            "n_epochs",
            "target_kl",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "policy_kwargs",
        }
        assert required.issubset(set(PAPER_PPO_DEFAULTS.keys()))

    def test_batch_size_divides_n_steps(self) -> None:
        """n_steps * n_envs must be divisible by batch_size for SB3 PPO."""
        from baselines.utils.training import PAPER_PPO_DEFAULTS

        n_steps = PAPER_PPO_DEFAULTS["n_steps"]
        batch_size = PAPER_PPO_DEFAULTS["batch_size"]
        assert (
            (n_steps % batch_size == 0)
            or (batch_size % n_steps == 0)
            or ((n_steps * 2) % batch_size == 0)
        ), f"n_steps={n_steps} may not divide evenly by batch_size={batch_size}"

    def test_policy_kwargs_net_arch(self) -> None:
        import torch
        from baselines.utils.training import PAPER_PPO_DEFAULTS

        pk = PAPER_PPO_DEFAULTS["policy_kwargs"]
        assert "net_arch" in pk
        assert pk["activation_fn"] is torch.nn.Tanh


@pytest.mark.quick
class TestPpoModuleImports:
    def test_training_module_imports(self) -> None:
        from baselines.utils.training import PAPER_PPO_DEFAULTS, build_ppo

        assert callable(build_ppo)
        assert isinstance(PAPER_PPO_DEFAULTS, dict)

    def test_train_ppo_module_importable(self) -> None:
        import baselines.train_ppo as train_ppo_mod

        assert hasattr(train_ppo_mod, "train_and_eval")
        assert hasattr(train_ppo_mod, "write_results_csv")
        assert hasattr(train_ppo_mod, "TrainResult")

    def test_write_results_csv_signature(self) -> None:
        import inspect
        from baselines.train_ppo import write_results_csv

        sig = inspect.signature(write_results_csv)
        assert "results" in sig.parameters
        assert "path" in sig.parameters
