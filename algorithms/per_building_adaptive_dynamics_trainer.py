"""
Per-building PPO trainer for the adaptive dynamics benchmark.

Trains a **separate** PPO agent on a single building from the test split.
Intended to be launched as a SLURM array job (one task per building) so that
all 100 test buildings are trained in parallel.

This serves as an upper-bound baseline: each agent is specialised to its own
building, so it should outperform any single generalised policy.
"""

import gc
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from algorithms.callbacks import make_adaptive_dynamics_callbacks
from building2building.training.sb3_utils import build_sb3_model, load_best_model
from building2building.baselines.wandb_utils import init_wandb_from_config
from building2building.benchmark.experiments.bm_adaptive_dynamics import log_returns_to_wandb
from building2building.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem
from building2building.api import make_env_from_hydra_config as make_env
from building2building.simulator.wrappers import NormalizeObservation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _create_env_for_split_index(
    config: OmegaConf,
    eplus_output_dir: str,
    split: str,
    split_index: int,
) -> gym.Env:
    """Create an EnergyPlus env for a specific building in the adaptive dynamics split."""
    output_path = Path(eplus_output_dir) / str(uuid.uuid4())
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating environment for %s split: split_index=%d", split, split_index)

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

    return make_env(
        config=OmegaConf.create(cfg_dict), eplus_output_dir=str(output_path)
    )


def _make_envs(
    config: OmegaConf,
    output_dir: Path,
    split: str,
    split_index: int,
    *,
    norm_obs: bool,
    norm_action: bool = False,
) -> tuple[VecNormalize, VecNormalize]:
    """Create train and eval vectorized environments for a single fixed building.

    Both train and eval use the SAME building (the one being trained on).
    This is correct for per-building training: we want to see how well the
    specialist performs on its own building during training.

    The final evaluation (after training) will test on all 100 test buildings.

    Note: Uses DummyVecEnv (not SubprocVecEnv) since there's only 1 environment.
    """

    def _wrap(env: gym.Env) -> gym.Env:
        if norm_action:
            env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        if norm_obs:
            env = NormalizeObservation(env)
        env = Monitor(env)
        return env

    def _make_train() -> gym.Env:
        env = _create_env_for_split_index(
            config,
            str(output_dir / "train_eplus_outputs"),
            split,
            split_index,
        )
        return _wrap(env)

    def _make_eval() -> gym.Env:
        env = _create_env_for_split_index(
            config,
            str(output_dir / "eval_eplus_outputs"),
            split,
            split_index,
        )
        return _wrap(env)

    raw_train_env = DummyVecEnv([_make_train])
    raw_eval_env = DummyVecEnv([_make_eval])

    # Wrap with VecNormalize for reward normalization only.
    # Observation normalization is already handled by NormalizeObservation.
    train_env = VecNormalize(raw_train_env, norm_obs=False, norm_reward=True)
    eval_env = VecNormalize(
        raw_eval_env, norm_obs=False, norm_reward=True, training=False
    )

    return train_env, eval_env


# ---------------------------------------------------------------------------
# Main trainer entry point
# ---------------------------------------------------------------------------


def per_building_adaptive_dynamics_trainer(
    config: OmegaConf,
    output_dir: Path,
    split: Literal["train", "test"],
    split_index: int,
) -> None:
    """Train a dedicated PPO agent on a single building.

    Parameters
    ----------
    config:
        Hydra configuration (merged with per_building_adaptive_dynamics.yaml).
    output_dir:
        Root output directory for this run.
    split:
        ``"train"`` or ``"test"``.
    split_index:
        Index into the split's row_ids list (0-based).
    """
    logger.info("=" * 80)
    logger.info("Per-Building Trainer  |  split=%s  split_index=%d", split, split_index)
    logger.info("=" * 80)
    logger.info("Output directory: %s", output_dir)

    # Tag the wandb run with the building index and batch ID for easy filtering.
    extra_tags = [f"split={split}", f"building_{split_index}"]

    # Add SLURM array job ID as a tag if available (for grouping runs from same batch)
    slurm_array_job_id = str(config.get("slurm_array_job_id", "none"))
    if slurm_array_job_id != "none":
        extra_tags.append(f"batch={slurm_array_job_id}")
        logger.info("SLURM array job ID: %s", slurm_array_job_id)

    wandb_run, started_here = init_wandb_from_config(
        config, run_dir=output_dir, extra_tags=extra_tags, sync_tensorboard=True
    )

    # Add batch ID to wandb config for easy filtering/grouping
    if wandb_run is not None and slurm_array_job_id != "none":
        try:
            wandb_run.config.update(
                {"slurm_array_job_id": slurm_array_job_id}, allow_val_change=True
            )
        except Exception as e:
            logger.warning("Failed to update wandb config with batch ID: %s", e)

    try:
        # Prepare IO dirs
        model_dir = output_dir / "models"
        log_dir = output_dir / "logs"
        tb_dir = output_dir / "tb"
        test_dir = output_dir / "test"
        for d in (model_dir, log_dir, tb_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)

        set_random_seed(config.seed)

        norm_obs: bool = config.env.normalize_obs
        norm_action: bool = config.env.get("normalize_action", False)

        # Create envs for this single building
        train_env, eval_env = _make_envs(
            config, output_dir, split, split_index,
            norm_obs=norm_obs, norm_action=norm_action,
        )

        logger.info(
            "Env: obs shape %s, action shape %s",
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

        # Load best model for evaluation
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
        # Benchmark on ALL test buildings using AdaptiveDynamicsProblem
        # (same as parameterized trainer for fair comparison)
        # ----------------------------------------------------------
        logger.info("Running benchmark on all 100 test buildings")

        cfg_dict_raw = OmegaConf.to_container(config, resolve=True)
        cfg_dict: dict[str, Any] = (
            dict(cfg_dict_raw) if isinstance(cfg_dict_raw, dict) else {}
        )
        cfg_dict["env"] = dict(cfg_dict.get("env", {}))
        cfg_dict["env"]["normalize_obs"] = False  # we normalize via wrapper

        def _eval_wrapper(env: gym.Env) -> gym.Env:
            if norm_action:
                env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
            if norm_obs:
                env = NormalizeObservation(env)
            return env

        problem = AdaptiveDynamicsProblem(
            split="test",  # Always evaluate on test split
            start=0,  # Start from first test building
            limit=0,  # 0 means all buildings in the split
            base_config=cfg_dict,
        )
        records = problem.run(
            best_model,
            output_dir=test_dir,
            env_wrapper=_eval_wrapper,
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
        logger.info(
            "Training and testing complete for building %d! (Evaluated on all 100 test buildings)",
            split_index,
        )
        logger.info("=" * 80)

    finally:
        if wandb_run is not None and started_here:
            wandb.finish()
