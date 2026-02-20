"""
Parameterized PPO trainer for the adaptive dynamics benchmark.

This trainer:
1. Trains on 900 buildings from the train split
2. Evaluates on 100 buildings from the test split
3. Uses building parameter augmentation for generalization
"""

import gc
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from algorithms.callbacks import make_adaptive_dynamics_callbacks
from algorithms.sb3_utils import build_sb3_model, load_best_model
from b2b.baselines.wandb_utils import init_wandb_from_config
from b2b.benchmark.experiments.bm_adaptive_dynamics import log_returns_to_wandb
from b2b.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem
from b2b.make_env import make_env
from b2b.simulator.wrappers import (
    AugmentObservationWithBuildingParams,
    NormalizeObservation,
    PadObservation,
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

    # Ensure env section is preserved
    if "env" not in cfg_dict:
        cfg_dict["env"] = {}

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
    wandb_prefix: str = "train",
):
    """
    Create an environment that resamples from the adaptive dynamics benchmark splits on each reset.

    Args:
        config: Hydra configuration
        eplus_output_dir: Directory for EnergyPlus outputs
        split: "train" or "test"
        split_indices: List of split indices to sample from. If None, uses all indices.
        wandb_prefix: Prefix for W&B log keys (e.g. ``"train"`` or ``"eval"``).

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
    env = ResampleBuildingOnResetWrapper(
        env_factory, split_indices, wandb_prefix=wandb_prefix
    )

    return env


def _apply_wrappers(
    env: gym.Env,
    *,
    target_obs_size: int | None,
    augment_params: bool,
    norm_obs: bool,
) -> gym.Env:
    """Apply the standard wrapper stack for adaptive dynamics envs.

    Order (inside → out):
        raw env → ResampleBuildingOnResetWrapper (already applied)
        → PadObservation (if target_obs_size is set)
        → AugmentObservationWithBuildingParams (if augment_params)
        → NormalizeObservation (if norm_obs)
        → Monitor (SB3 episode tracking for rollout/ metrics)
    """
    if target_obs_size is not None:
        env = PadObservation(env, target_size=target_obs_size)
    if augment_params:
        env = AugmentObservationWithBuildingParams(env)
    if norm_obs:
        env = NormalizeObservation(env)
    env = Monitor(env)
    return env


def _make_single_wrapped_env(
    config: OmegaConf,
    eplus_output_dir: str,
    split: str,
    split_indices: list[int],
    *,
    target_obs_size: int | None,
    augment_params: bool,
    norm_obs: bool,
    wandb_prefix: str,
) -> gym.Env:
    """Create one fully-wrapped adaptive dynamics env.

    The wrapper stack (inside → out) is:
        raw EnergyPlus env
        → ResampleBuildingOnResetWrapper
        → PadObservation
        → AugmentObservationWithBuildingParams
        → NormalizeObservation
    """
    env = make_adaptive_dynamics_env(
        config=config,
        eplus_output_dir=eplus_output_dir,
        split=split,
        split_indices=split_indices,
        wandb_prefix=wandb_prefix,
    )
    return _apply_wrappers(
        env,
        target_obs_size=target_obs_size,
        augment_params=augment_params,
        norm_obs=norm_obs,
    )


def _make_adaptive_dynamics_envs(
    config: OmegaConf,
    output_dir: Path,
    *,
    target_obs_size: int | None,
    augment_params: bool,
    norm_obs: bool,
) -> tuple[VecNormalize, VecNormalize]:
    """Create vectorized training and single evaluation environments.

    Training uses ``n_train_envs`` parallel environments (each independently
    resampling buildings via ``ResampleBuildingOnResetWrapper``) so that each
    PPO rollout buffer contains experiences from many diverse buildings.

    Evaluation uses a single environment for deterministic benchmarking.

    Buildings in the split may have different raw observation sizes (e.g. 8, 9,
    or 10 dims).  When *target_obs_size* is set the observations are
    zero-padded to that fixed size so that SB3's fixed-shape buffers work.
    """
    n_train_envs: int = int(config.training.num_train_envs)

    # Load splits to get sizes
    splits = HydroQuebecRowIdSplits.load_from_action_space_2_zone_1()
    train_indices = list(range(len(splits.train_row_ids)))
    test_indices = list(range(len(splits.test_row_ids)))

    logger.info(
        "Adaptive dynamics benchmark: %d train buildings, %d test buildings",
        len(train_indices),
        len(test_indices),
    )
    logger.info("Using %d parallel training environments", n_train_envs)
    if target_obs_size is not None:
        logger.info("Padding raw observations to fixed size %d", target_obs_size)

    # Shared keyword arguments for _make_single_wrapped_env
    wrapper_kwargs = {
        "target_obs_size": target_obs_size,
        "augment_params": augment_params,
        "norm_obs": norm_obs,
    }

    # --- Vectorized training environments ---
    def _make_train_env(env_idx: int) -> gym.Env:
        return _make_single_wrapped_env(
            config=config,
            eplus_output_dir=str(output_dir / f"train_eplus_outputs_{env_idx}"),
            split="train",
            split_indices=train_indices,
            wandb_prefix="train",
            **wrapper_kwargs,
        )

    # Use SubprocVecEnv for parallel training (runs each env in separate process)
    raw_train_env = SubprocVecEnv(
        [lambda idx=i: _make_train_env(idx) for i in range(n_train_envs)]
    )

    # --- Single evaluation environment (wrapped in DummyVecEnv for SB3) ---
    def _make_eval_env() -> gym.Env:
        return _make_single_wrapped_env(
            config=config,
            eplus_output_dir=str(output_dir / "eval_eplus_outputs"),
            split="test",
            split_indices=test_indices,
            wandb_prefix="eval",
            **wrapper_kwargs,
        )

    # Keep eval as DummyVecEnv (single env, no benefit from multiprocessing)
    raw_eval_env = DummyVecEnv([_make_eval_env])

    # Wrap with VecNormalize for reward normalization only.
    # Observation normalization is already handled by NormalizeObservation.
    train_env = VecNormalize(raw_train_env, norm_obs=False, norm_reward=True)
    eval_env = VecNormalize(
        raw_eval_env, norm_obs=False, norm_reward=True, training=False
    )

    return train_env, eval_env


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
    wandb_run, started_here = init_wandb_from_config(
        config, run_dir=output_dir, sync_tensorboard=True
    )

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

        # Read wrapper config once — used for both training envs and
        # the post-training benchmark.
        norm_obs: bool = config.env.normalize_obs
        augment_params: bool = config.env.get("augment_building_params", True)
        target_obs_size: int | None = config.env.get("target_obs_size", None)

        # Create adaptive dynamics environments
        logger.info(
            "Creating adaptive dynamics environments with building parameter augmentation"
        )
        train_env, eval_env = _make_adaptive_dynamics_envs(
            config,
            output_dir,
            target_obs_size=target_obs_size,
            augment_params=augment_params,
            norm_obs=norm_obs,
        )

        # Log observation space info
        logger.info(
            "Train VecEnv: %d envs, obs shape %s, action shape %s",
            train_env.num_envs,
            train_env.observation_space.shape,
            train_env.action_space.shape,
        )

        # Build model and callbacks
        callbacks = make_adaptive_dynamics_callbacks(
            config, eval_env, model_dir, log_dir
        )
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

        # Close training environments to free memory before benchmark.
        # Each env keeps an EnergyPlus process alive; running the benchmark
        # with them still open can cause OOM kills.
        logger.info("Closing training environments to free memory")
        try:
            train_env.close()
        except Exception as exc:
            logger.warning("Error closing train_env: %s", exc)
        try:
            eval_env.close()
        except Exception as exc:
            logger.warning("Error closing eval_env: %s", exc)
        del train_env, eval_env, model
        gc.collect()

        # ----------------------------------------------------------
        # Benchmark on the full test split using AdaptiveDynamicsProblem
        # ----------------------------------------------------------
        logger.info("Running adaptive dynamics benchmark on test split")

        cfg_dict_raw = OmegaConf.to_container(config, resolve=True)
        cfg_dict: dict[str, Any] = (
            dict(cfg_dict_raw) if isinstance(cfg_dict_raw, dict) else {}
        )
        # The benchmark creates raw envs; we must apply the same
        # wrapper stack so the model sees the expected obs shape.
        cfg_dict["env"] = dict(cfg_dict.get("env", {}))
        cfg_dict["env"]["normalize_obs"] = False  # we normalise via wrapper

        problem = AdaptiveDynamicsProblem(
            split="test",
            base_config=cfg_dict,
        )
        records = problem.run(
            best_model,
            output_dir=test_dir,
            env_wrapper=lambda env: _apply_wrappers(
                env,
                target_obs_size=target_obs_size,
                augment_params=augment_params,
                norm_obs=norm_obs,
            ),
        )

        if wandb_run is not None:
            returns: list[float] = [
                float(r.episode_result.total_reward)
                for r in records
                if r.episode_result is not None
            ]
            log_returns_to_wandb(returns=returns)

            # Upload detailed per-building results to wandb for analysis
            jsonl_path = test_dir / "adaptive_dynamics_results.jsonl"
            if jsonl_path.exists():
                try:
                    artifact = wandb.Artifact(
                        name="test_results",
                        type="evaluation",
                        description="Detailed per-building evaluation results",
                    )
                    artifact.add_file(str(jsonl_path))
                    wandb_run.log_artifact(artifact)
                    logger.info("Uploaded test results to wandb artifact")
                except Exception as e:
                    logger.warning("Failed to upload test results artifact: %s", e)

        logger.info("=" * 80)
        logger.info("Training and testing complete!")
        logger.info("=" * 80)

    finally:
        if wandb_run is not None and started_here:
            wandb.finish()
