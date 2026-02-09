"""
Parameterized PPO trainer for the adaptive dynamics benchmark.

This trainer:
1. Trains on 900 buildings from the train split
2. Evaluates on 100 buildings from the test split
3. Uses building parameter augmentation for generalization
"""

import importlib
import inspect
import logging
import random
import shutil
import uuid
from pathlib import Path

import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

from b2b.baselines.test import test_policy
from b2b.baselines.utils import log_test_dir_graphs_wandb
from b2b.baselines.wandb_utils import init_wandb_from_config
from b2b.make_env import make_env
from b2b.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
    ResampleBuildingOnResetWrapper,
)
from b2b.utils import HydroQuebecRowIdSplits

logger = logging.getLogger(__name__)


def _create_env_for_split_index(
    config, eplus_output_dir: str, split: str, split_index: int
):
    """
    Create an environment for a specific building from the adaptive dynamics benchmark.

    Args:
        config: Hydra configuration
        eplus_output_dir: Directory for EnergyPlus outputs
        split: "train" or "test"
        split_index: Index into the split's row_ids list

    Returns:
        gym.Env: EnergyPlus environment
    """
    # Create a unique directory for this environment instance
    output_path = Path(eplus_output_dir) / str(uuid.uuid4())

    # Clean up if it already exists
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating environment for %s split: split_index=%d", split, split_index)

    # Create config with selection enabled
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    bldg_section = cfg_dict.setdefault("bldg", {})
    if not isinstance(bldg_section, dict):
        bldg_section = {}
        cfg_dict["bldg"] = bldg_section

    bldg_section["selection"] = {
        "enabled": True,
        "split": split,
        "index": split_index,
    }

    # Create environment
    env = make_env(config=OmegaConf.create(cfg_dict), eplus_output_dir=str(output_path))

    return env


def make_adaptive_dynamics_env(
    config,
    eplus_output_dir: str,
    split: str = "train",
    split_indices: list[int] | None = None,
):
    """
    Create an environment that resamples from the adaptive dynamics benchmark splits on each reset.

    Args:
        config: Hydra configuration
        eplus_output_dir: Directory for EnergyPlus outputs
        split: "train" or "test"
        split_indices: List of split indices to sample from. If None, uses all indices.

    Returns:
        gym.Env: EnergyPlus environment wrapped with ResampleBuildingOnResetWrapper
    """
    # Build available indices from the split
    if split_indices is None:
        splits = HydroQuebecRowIdSplits.load_from_action_space_2_zone_1()
        row_ids = splits.train_row_ids if split == "train" else splits.test_row_ids
        split_indices = list(range(len(row_ids)))

    if not split_indices:
        raise ValueError("No buildings available for split %r" % split)

    # Create a factory function that creates environments for specific split indices
    def env_factory(split_index: int):
        return _create_env_for_split_index(config, eplus_output_dir, split, split_index)

    # Wrap with resampling wrapper
    env = ResampleBuildingOnResetWrapper(env_factory, split_indices)

    return env


def _make_adaptive_dynamics_envs(config: OmegaConf, output_dir: Path):
    """
    Create training and evaluation environments for adaptive dynamics benchmark.

    Note: We don't use vectorized environments because buildings have different
    observation space sizes, which would cause shape mismatches.
    """
    # Read required config
    norm_obs = config.env.normalize_obs
    augment_params = config.env.get("augment_building_params", True)

    # Load splits to get sizes
    splits = HydroQuebecRowIdSplits.load_from_action_space_2_zone_1()
    n_train = len(splits.train_row_ids)
    n_test = len(splits.test_row_ids)

    logger.info(
        "Adaptive dynamics benchmark: %d train buildings, %d test buildings",
        n_train,
        n_test,
    )
    logger.info(
        "Note: Not using vectorized environments due to variable observation space sizes"
    )

    # Create single training environment
    train_env = make_adaptive_dynamics_env(
        config=config,
        eplus_output_dir=str(output_dir / "train_eplus_outputs"),
        split="train",
        split_indices=None,
    )

    # Apply wrappers
    if augment_params:
        train_env = AugmentObservationWithBuildingParams(train_env)
    if norm_obs:
        train_env = NormalizeObservation(train_env)

    # Create single evaluation environment
    eval_env = make_adaptive_dynamics_env(
        config=config,
        eplus_output_dir=str(output_dir / "eval_eplus_outputs"),
        split="test",
        split_indices=None,
    )

    # Apply wrappers
    if augment_params:
        eval_env = AugmentObservationWithBuildingParams(eval_env)
    if norm_obs:
        eval_env = NormalizeObservation(eval_env)

    return train_env, eval_env


def _make_callbacks(config: OmegaConf, eval_env, model_dir: Path, log_dir: Path):
    wandb_cb = WandbCallback(
        gradient_save_freq=config.training.cb_gradient_save_freq,
        model_save_path=str(model_dir),
        verbose=2,
    )
    eval_cb = EvalCallback(
        eval_env,
        log_path=str(log_dir),
        eval_freq=config.training.eval_freq,
        best_model_save_path=str(model_dir),
        n_eval_episodes=config.training.eval_episodes,
        deterministic=True,
    )
    return CallbackList([eval_cb, wandb_cb])


def _build_sb3_model(config: OmegaConf, train_env, tb_dir: Path):
    """Build a Stable Baselines 3 model based on the config."""

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

    kwargs["tensorboard_log"] = str(tb_dir)
    policy = config.policy.policy_type

    model = algo_cls(policy, train_env, **kwargs)

    return model


def _load_best_model(config: OmegaConf, model_dir: Path):
    """Load the best model saved during training."""
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


def parameterized_adaptive_dynamics_trainer(config: OmegaConf, output_dir: Path):
    """
    Train a parameterized PPO policy on the adaptive dynamics benchmark.

    Training: 900 buildings from train split
    Evaluation: 100 buildings from test split
    """
    logger.info("=" * 80)
    logger.info("Parameterized Adaptive Dynamics Trainer")
    logger.info("=" * 80)
    logger.info("Output directory: %s", output_dir)

    # Initialize WandB
    wandb_run, started_here = init_wandb_from_config(config, run_dir=output_dir)

    try:
        # Prepare IO dirs
        model_dir = output_dir / "models"
        log_dir = output_dir / "logs"
        tb_dir = output_dir / "tb"
        test_dir = output_dir / "test"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        set_random_seed(config.seed)

        # Create adaptive dynamics environments
        logger.info(
            "Creating adaptive dynamics environments with building parameter augmentation"
        )
        train_env, eval_env = _make_adaptive_dynamics_envs(config, output_dir)

        # Log observation space info
        logger.info("Observation space shape: %s", train_env.observation_space.shape)
        logger.info("Action space shape: %s", train_env.action_space.shape)

        # Build model and callbacks
        callbacks = _make_callbacks(config, eval_env, model_dir, log_dir)
        model = _build_sb3_model(config, train_env, tb_dir)

        # Train
        total_timesteps = int(config.training.total_timesteps)
        logger.info("Starting training for %d timesteps", total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        logger.info("Training complete")

        # Save final model
        final_model_path = model_dir / "final_model.zip"
        model.save(str(final_model_path))
        logger.info("Saved final model to %s", final_model_path)

        # Load best model for testing
        best_model = _load_best_model(config, model_dir)
        if best_model is None:
            logger.warning("Could not load best model, using final model for testing")
            best_model = model

        # Test on multiple test buildings
        logger.info("Testing best model on test split buildings")
        test_policy(config, best_model, output_dir)

        # Log test results to WandB
        if wandb_run is not None:
            log_test_dir_graphs_wandb(test_dir)

        logger.info("=" * 80)
        logger.info("Training and testing complete!")
        logger.info("=" * 80)

    finally:
        if wandb_run is not None and started_here:
            wandb.finish()
