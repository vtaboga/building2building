"""
Shared callback classes for multi-building RL training.

This module contains callback classes used across different trainers to log
diagnostics, evaluation metrics, and trajectory information to Weights & Biases.
"""

import logging
from pathlib import Path

import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback

from b2b.baselines.callbacks import TrainingEpisodeRewardCallback

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
                        }
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

            wandb.log(log_dict)

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
                }
            )

            if self.verbose >= 1:
                logger.info(
                    "Trajectory snippet logged: %d steps, mean_reward=%.3f",
                    len(snippet),
                    np.mean([s["reward"] for s in snippet]),
                )

        # Clear buffer to avoid memory buildup
        self.trajectory_buffer = []


def make_adaptive_dynamics_callbacks(
    config: OmegaConf, eval_env, model_dir: Path, log_dir: Path
) -> CallbackList:
    """
    Create the standard callback list for adaptive dynamics trainers.

    Args:
        config: Hydra configuration object
        eval_env: Evaluation environment
        model_dir: Directory to save model checkpoints
        log_dir: Directory to save logs

    Returns:
        CallbackList with all configured callbacks
    """
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

    train_ep_cb = TrainingEpisodeRewardCallback()

    return CallbackList(
        [
            eval_cb,
            eval_histogram_cb,
            action_dist_cb,
            trajectory_cb,
            train_ep_cb,
            wandb_cb,
        ]
    )
