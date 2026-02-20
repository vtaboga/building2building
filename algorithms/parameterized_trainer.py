"""
Parameterized online trainer for multi-building RL.

This trainer augments observations with building-specific parameters,
allowing a single policy to generalize across multiple buildings.
"""

import importlib
import inspect
import logging
from pathlib import Path

import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

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
    """Load the best saved SB3 model if available, else return None."""
    algo_name = config.policy.algorithm
    best_path = model_dir / "best_model.zip"
    if not best_path.exists() or not algo_name:
        return None
    algo_upper = str(algo_name).upper()
    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algo_name}.{algo_name}")
            algo_cls = getattr(module, algo_upper)
            return algo_cls.load(str(best_path))
        except Exception:
            continue
    return None


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
