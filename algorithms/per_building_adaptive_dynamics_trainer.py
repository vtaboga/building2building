"""
Per-building PPO trainer for the adaptive dynamics benchmark.

Trains a **separate** PPO agent on a single building from the test split.
Intended to be launched as a SLURM array job (one task per building) so that
all 100 test buildings are trained in parallel.

This serves as an upper-bound baseline: each agent is specialised to its own
building, so it should outperform any single generalised policy.
"""

import gc
import importlib
import inspect
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from b2b.baselines.wandb_utils import init_wandb_from_config
from b2b.benchmark.experiments.bm_adaptive_dynamics import log_returns_to_wandb
from b2b.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem
from b2b.make_env import make_env
from b2b.simulator.wrappers import NormalizeObservation

logger = logging.getLogger(__name__)


class EvalHistogramCallback(BaseCallback):
    """Log eval reward histogram to wandb after each evaluation."""

    def __init__(self, eval_callback: EvalCallback, verbose: int = 0):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        # Check if EvalCallback just ran an evaluation
        if (
            len(self.eval_callback.evaluations_timesteps) > 0
            and self.eval_callback.evaluations_timesteps[-1] > self._last_eval_timestep
        ):
            # New evaluation just completed
            self._last_eval_timestep = self.eval_callback.evaluations_timesteps[-1]

            # Get the most recent eval episode rewards
            if len(self.eval_callback.evaluations_results) > 0:
                episode_rewards = self.eval_callback.evaluations_results[-1]

                # Log histogram to wandb
                if wandb.run is not None:
                    wandb.log(
                        {
                            "eval/reward_histogram": wandb.Histogram(episode_rewards),
                            "eval/reward_mean": np.mean(episode_rewards),
                            "eval/reward_std": np.std(episode_rewards),
                            "eval/reward_min": np.min(episode_rewards),
                            "eval/reward_max": np.max(episode_rewards),
                        },
                        step=self.num_timesteps,
                    )

                    if self.verbose >= 1:
                        logger.info(
                            "Eval histogram logged: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
                            np.mean(episode_rewards),
                            np.std(episode_rewards),
                            np.min(episode_rewards),
                            np.max(episode_rewards),
                        )

        return True


class ActionDistributionCallback(BaseCallback):
    """Log action distribution statistics during rollouts to detect trivial policies."""

    def __init__(self, log_freq: int = 2048, verbose: int = 0):
        """
        Args:
            log_freq: How often to log action statistics (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.actions_buffer = []
        self._last_log_timestep = 0

    def _on_step(self) -> bool:
        # Collect actions from the current step
        # For vectorized envs, self.locals contains 'actions' from the last step
        if "actions" in self.locals:
            actions = self.locals["actions"]
            # Convert to numpy array if needed
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            self.actions_buffer.append(actions)

        # Log statistics every log_freq steps
        if self.num_timesteps - self._last_log_timestep >= self.log_freq:
            if len(self.actions_buffer) > 0:
                self._log_action_stats()
                self.actions_buffer = []
                self._last_log_timestep = self.num_timesteps

        return True

    def _log_action_stats(self):
        """Compute and log action distribution statistics."""
        if len(self.actions_buffer) == 0:
            return

        # Stack all actions: shape (num_steps, num_envs, action_dim) or (num_steps, action_dim)
        all_actions = np.array(self.actions_buffer)

        # Flatten to (num_samples, action_dim)
        if all_actions.ndim == 3:
            # Vectorized env case: (num_steps, num_envs, action_dim)
            all_actions = all_actions.reshape(-1, all_actions.shape[-1])
        elif all_actions.ndim == 2:
            # Single env case: (num_steps, action_dim)
            pass
        else:
            logger.warning(f"Unexpected action shape: {all_actions.shape}")
            return

        # Compute statistics per action dimension
        action_means = np.mean(all_actions, axis=0)
        action_stds = np.std(all_actions, axis=0)
        action_mins = np.min(all_actions, axis=0)
        action_maxs = np.max(all_actions, axis=0)

        # Overall statistics
        overall_mean = np.mean(all_actions)
        overall_std = np.std(all_actions)

        # Log to wandb
        if wandb.run is not None:
            log_dict = {
                "rollout/action_mean": overall_mean,
                "rollout/action_std": overall_std,
                "rollout/action_mean_of_stds": np.mean(action_stds),
            }

            # Log per-dimension statistics (for first few dimensions to avoid clutter)
            max_dims_to_log = min(10, len(action_means))
            for i in range(max_dims_to_log):
                log_dict[f"rollout/action_dim_{i}_mean"] = action_means[i]
                log_dict[f"rollout/action_dim_{i}_std"] = action_stds[i]

            # Log histograms for each action dimension
            for i in range(max_dims_to_log):
                log_dict[f"rollout/action_dim_{i}_histogram"] = wandb.Histogram(
                    all_actions[:, i]
                )

            wandb.log(log_dict, step=self.num_timesteps)

            if self.verbose >= 1:
                logger.info(
                    "Action stats logged: overall_mean=%.3f, overall_std=%.3f, mean_of_stds=%.3f",
                    overall_mean,
                    overall_std,
                    np.mean(action_stds),
                )


class TrajectorySnippetCallback(BaseCallback):
    """Log a short trajectory snippet after every PPO update."""

    def __init__(self, snippet_length: int = 10, verbose: int = 0):
        """
        Args:
            snippet_length: Number of timesteps to include in each snippet
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.snippet_length = snippet_length
        self.trajectory_buffer = []

    def _on_step(self) -> bool:
        """Collect trajectory data during rollout."""
        # Collect data from the current step
        if "obs_tensor" in self.locals or "new_obs" in self.locals:
            obs = self.locals.get("new_obs", self.locals.get("obs_tensor"))
            actions = self.locals.get("actions")
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")

            # Store trajectory step (only from first env to keep it simple)
            if obs is not None and actions is not None and rewards is not None:
                # For vectorized envs, take first environment
                if isinstance(obs, np.ndarray) and obs.ndim > 1:
                    obs = obs[0]
                if isinstance(actions, np.ndarray) and actions.ndim > 1:
                    actions = actions[0]
                if isinstance(rewards, np.ndarray):
                    rewards = rewards[0] if rewards.ndim > 0 else float(rewards)
                if isinstance(dones, np.ndarray):
                    dones = dones[0] if dones.ndim > 0 else bool(dones)

                self.trajectory_buffer.append(
                    {
                        "obs": np.array(obs).copy(),
                        "action": np.array(actions).copy(),
                        "reward": float(rewards),
                        "done": bool(dones),
                    }
                )

        return True

    def _on_rollout_end(self) -> None:
        """Log trajectory snippet after PPO update."""
        if len(self.trajectory_buffer) == 0:
            return

        # Take the last snippet_length steps
        snippet = self.trajectory_buffer[-self.snippet_length :]

        if wandb.run is not None:
            # Create a table for the trajectory snippet
            columns = ["step", "reward", "done"]

            # Add action columns
            if len(snippet) > 0 and "action" in snippet[0]:
                action_dim = len(snippet[0]["action"])
                for i in range(action_dim):
                    columns.append(f"action_{i}")

            # Add observation columns (limit to first 10 dims to avoid clutter)
            if len(snippet) > 0 and "obs" in snippet[0]:
                obs_dim = min(10, len(snippet[0]["obs"]))
                for i in range(obs_dim):
                    columns.append(f"obs_{i}")

            table = wandb.Table(columns=columns)

            for step_idx, step_data in enumerate(snippet):
                row = [
                    step_idx,
                    step_data["reward"],
                    step_data["done"],
                ]

                # Add actions
                for action_val in step_data["action"]:
                    row.append(float(action_val))

                # Add observations (first 10 dims)
                obs_to_log = step_data["obs"][:obs_dim]
                for obs_val in obs_to_log:
                    row.append(float(obs_val))

                table.add_data(*row)

            wandb.log(
                {
                    "trajectory/snippet": table,
                    "trajectory/snippet_mean_reward": np.mean(
                        [s["reward"] for s in snippet]
                    ),
                },
                step=self.num_timesteps,
            )

            if self.verbose >= 1:
                logger.info(
                    "Trajectory snippet logged: %d steps, mean_reward=%.3f",
                    len(snippet),
                    np.mean([s["reward"] for s in snippet]),
                )

        # Clear buffer to avoid memory buildup
        self.trajectory_buffer = []


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
) -> tuple[VecNormalize, VecNormalize]:
    """Create train and eval vectorized environments for a single fixed building.

    Both train and eval use the SAME building (the one being trained on).
    This is correct for per-building training: we want to see how well the
    specialist performs on its own building during training.

    The final evaluation (after training) will test on all 100 test buildings.

    Note: Uses DummyVecEnv (not SubprocVecEnv) since there's only 1 environment.
    """

    def _wrap(env: gym.Env) -> gym.Env:
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
# SB3 model helpers (same logic as parameterized trainer)
# ---------------------------------------------------------------------------


def _make_callbacks(
    config: OmegaConf, eval_env: DummyVecEnv, model_dir: Path, log_dir: Path
) -> CallbackList:
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
    # Add histogram logging callback
    eval_histogram_cb = EvalHistogramCallback(eval_cb, verbose=1)

    # Add action distribution logging callback
    # Log every n_steps (same as PPO rollout buffer size)
    action_log_freq = config.policy.get("n_steps", 2048)
    action_dist_cb = ActionDistributionCallback(log_freq=action_log_freq, verbose=1)

    # Add trajectory snippet logging callback
    snippet_length = config.training.get("trajectory_snippet_length", 10)
    trajectory_cb = TrajectorySnippetCallback(snippet_length=snippet_length, verbose=1)

    return CallbackList(
        [eval_cb, eval_histogram_cb, action_dist_cb, trajectory_cb, wandb_cb]
    )


def _build_sb3_model(config: OmegaConf, train_env: DummyVecEnv, tb_dir: Path):
    """Build a Stable Baselines 3 model based on the config."""
    params = OmegaConf.to_container(config.policy, resolve=True)
    algo_name = config.policy.algorithm

    if not algo_name:
        raise ValueError("The key algorithm in policy config is missing.")

    algo_upper = str(algo_name).upper()
    algo_cls = None

    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algo_name}.{algo_name}")
            algo_cls = getattr(module, algo_upper)
            break
        except (ModuleNotFoundError, AttributeError):
            continue

    if algo_cls is None:
        raise ValueError(
            f"Algorithm '{algo_name}' not found. "
            "Expected it to exist as "
            f"stable_baselines3.{algo_name}.{algo_name}.{algo_upper} "
            f"or sb3_contrib.{algo_name}.{algo_name}.{algo_upper}."
        )

    sig = inspect.signature(algo_cls.__init__)
    valid_params = {
        k for k in sig.parameters.keys() if k not in ["self", "env", "policy"]
    }
    kwargs = {k: v for k, v in params.items() if k in valid_params}
    kwargs["tensorboard_log"] = str(tb_dir)

    return algo_cls(config.policy.policy_type, train_env, **kwargs)


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

        # Create envs for this single building
        train_env, eval_env = _make_envs(
            config, output_dir, split, split_index, norm_obs=norm_obs
        )

        logger.info(
            "Env: obs shape %s, action shape %s",
            train_env.observation_space.shape,
            train_env.action_space.shape,
        )

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

        # Load best model for evaluation
        best_model = _load_best_model(config, model_dir)
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
