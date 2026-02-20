"""
Parameterized online trainer for multi-building RL.

This trainer augments observations with building-specific parameters,
allowing a single policy to generalize across multiple buildings.
"""

import logging
from pathlib import Path

import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

from algorithms.sb3_utils import build_sb3_model, load_best_model
from b2b.baselines.callbacks import TrainingEpisodeRewardCallback
from b2b.baselines.test import test_policy
from b2b.baselines.utils import log_test_dir_graphs_wandb
from b2b.baselines.wandb_utils import init_wandb_from_config
from b2b.make_env import make_env
from b2b.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
)

logger = logging.getLogger(__name__)


def _make_parameterized_envs(config: OmegaConf, output_dir: Path):
    """
    Create training and evaluation environments with building parameter augmentation.

    This differs from the standard trainer by adding building parameters to observations,
    enabling a single policy to work across multiple buildings.
    """
    # Read required config without providing defaults; raise if missing
    norm_obs = config.env.normalize_obs  # expect under env
    augment_params = config.env.get("augment_building_params", True)

    def wrapper_fn(env):
        # First augment with building parameters (if enabled)
        if augment_params:
            env = AugmentObservationWithBuildingParams(env)
        # Then normalize observations (if enabled)
        if norm_obs:
            env = NormalizeObservation(env)
        return env

    num_envs = int(config.training.num_train_envs)
    train_env = make_vec_env(
        make_env,
        n_envs=num_envs,
        env_kwargs={
            "config": config,
            "eplus_output_dir": str(output_dir / "train_eplus_outputs"),
        },
        wrapper_class=wrapper_fn,
    )

    # For evaluation, we typically use a single environment
    eval_env = make_vec_env(
        make_env,
        n_envs=1,
        env_kwargs={
            "config": config,
            "eplus_output_dir": str(output_dir / "eval_eplus_outputs"),
        },
        wrapper_class=wrapper_fn,
    )

    return train_env, eval_env


def _make_callbacks(config: OmegaConf, eval_env, model_dir: Path, log_dir: Path):
    """Create training callbacks for WandB logging and evaluation."""
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
    train_ep_cb = TrainingEpisodeRewardCallback()
    return CallbackList([eval_cb, train_ep_cb, wandb_cb])


def parameterized_trainer(config: OmegaConf, output_dir: Path):
    """
    Train a parameterized policy that can generalize across multiple buildings.

    This trainer augments observations with building-specific parameters
    (area, warmup_phases, num_actuators) so a single policy can learn to
    control different buildings.

    Args:
        config: Hydra configuration
        output_dir: Directory for outputs (models, logs, etc.)
    """
    # Stable reference to the repository root
    repo_root = Path(__file__).resolve().parents[1]

    wandb_run, _started_here = init_wandb_from_config(
        config,
        run_dir=output_dir,
        log_code_root=repo_root,
        sync_tensorboard=True,
    )

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

    # Create parameterized environments
    logger.info(
        "Creating parameterized environments with building parameter augmentation"
    )
    train_env, eval_env = _make_parameterized_envs(config, output_dir)

    # Log observation space info
    logger.info("Observation space shape: %s", train_env.observation_space.shape)
    logger.info("Action space shape: %s", train_env.action_space.shape)

    # Build model and callbacks
    callbacks = _make_callbacks(config, eval_env, model_dir, log_dir)
    model = build_sb3_model(config, train_env, tb_dir)

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
    best_model = load_best_model(config, model_dir)
    if best_model is None:
        logger.warning("Could not load best model, using final model for testing")
        best_model = model

    # Test the best model
    logger.info("Running test rollouts with best model")
    test_policy(config, best_model, output_dir)

    # Log test graphs to WandB
    if wandb_run is not None:
        logger.info("Logging test graphs to WandB")
        log_test_dir_graphs_wandb(output_dir / "test")

    # Clean up
    train_env.close()
    eval_env.close()

    if wandb_run is not None:
        wandb.finish()

    logger.info("Parameterized training complete. Outputs saved to %s", output_dir)
