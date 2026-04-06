"""
Shared utilities for working with Stable Baselines 3 (SB3) models.

This module provides common functions for building and loading SB3 models
across different trainers.
"""

import importlib
import inspect
import logging
from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def build_sb3_model(config: OmegaConf, train_env, tb_dir: Path):
    """
    Build a Stable Baselines 3 model based on the config.

    Args:
        config: Hydra configuration object containing policy settings
        train_env: Training environment (can be vectorized)
        tb_dir: Directory for TensorBoard logs

    Returns:
        Initialized SB3 model instance

    Raises:
        ValueError: If algorithm is not specified or not found
    """
    params = OmegaConf.to_container(config.policy, resolve=True)
    algo_name = config.policy.algorithm

    if not algo_name:
        raise ValueError("The key algorithm in policy config is missing.")

    # Resolve algorithm class from SB3 or SB3-Contrib (e.g. TRPO).
    algo_upper = str(algo_name).upper()
    module = None
    algo_cls = None

    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algo_name}.{algo_name}")
            algo_cls = getattr(module, algo_upper)
            break
        except ModuleNotFoundError:
            continue
        except AttributeError:
            continue

    if algo_cls is None:
        raise ValueError(
            f"Algorithm '{algo_name}' not found. "
            "Expected it to exist as "
            f"stable_baselines3.{algo_name}.{algo_name}.{algo_upper} "
            f"or sb3_contrib.{algo_name}.{algo_name}.{algo_upper}."
        )

    # pass only valid arguments
    sig = inspect.signature(algo_cls.__init__)
    valid_params = {
        k for k in sig.parameters.keys() if k not in ["self", "env", "policy"]
    }
    kwargs = {k: v for k, v in params.items() if k in valid_params}

    # Resolve activation_fn string to torch.nn class if needed.
    pk = kwargs.get("policy_kwargs")
    if isinstance(pk, dict) and isinstance(pk.get("activation_fn"), str):
        pk["activation_fn"] = getattr(nn, pk["activation_fn"])

    kwargs["tensorboard_log"] = str(tb_dir)
    policy = config.policy.policy_type

    model = algo_cls(policy, train_env, **kwargs)

    return model


def load_best_model(config: OmegaConf, model_dir: Path):
    """
    Load the best model saved during training.

    Args:
        config: Hydra configuration object containing policy settings
        model_dir: Directory where the best model was saved

    Returns:
        Loaded SB3 model instance, or None if loading failed
    """
    best_model_path = model_dir / "best_model.zip"

    if not best_model_path.exists():
        logger.warning("Best model not found at %s", best_model_path)
        return None

    algo_name = config.policy.algorithm
    algo_upper = str(algo_name).upper()

    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algo_name}.{algo_name}")
            algo_cls = getattr(module, algo_upper)
            model = algo_cls.load(str(best_model_path))
            logger.info("Loaded best model from %s", best_model_path)
            return model
        except (ModuleNotFoundError, AttributeError):
            continue

    logger.error("Could not load model: algorithm %r not found", algo_name)
    return None
