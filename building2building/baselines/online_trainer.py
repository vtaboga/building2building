import logging
from pathlib import Path

import gymnasium as gym
from omegaconf import OmegaConf
from wandb.integration.sb3 import WandbCallback
import wandb

from building2building.training.sb3_utils import build_sb3_model, load_best_model
from building2building.baselines.callbacks import TrainingEpisodeRewardCallback
from building2building.baselines.test import test_policy
from building2building.baselines.utils import make_dummy_vec_env, make_env, log_test_dir_graphs_wandb
from building2building.baselines.wandb_utils import init_wandb_from_config
from building2building.simulator.wrappers import NormalizeObservation
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

logger = logging.getLogger(__name__)


def _make_envs(config: OmegaConf, output_dir: Path):

    # Read required config without providing defaults; raise if missing
    norm_obs = config.env.normalize_obs  # expect under env
    norm_action = getattr(config.env, "normalize_action", False)

    def wrapper_fn(env):
        if norm_action:
            env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        if norm_obs:
            env = NormalizeObservation(env)
        return env

    num_envs = int(config.training.num_train_envs)
    vec_cls = SubprocVecEnv if num_envs > 1 else None
    train_env = make_vec_env(
        make_env,
        n_envs=num_envs,
        env_kwargs={
            "config": config,
            "eplus_output_dir": str(output_dir / "train_eplus_outputs"),
        },
        wrapper_class=wrapper_fn,
        vec_env_cls=vec_cls,
    )
    eval_env = make_dummy_vec_env(
        config=config,
        eplus_output_dir=str(output_dir / "eval_eplus_outputs"),
        wrapper_fn=wrapper_fn,
    )
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
    train_ep_cb = TrainingEpisodeRewardCallback()
    return CallbackList([eval_cb, train_ep_cb, wandb_cb])


def online_trainer(config: OmegaConf, output_dir: Path):

    # Stable reference to the repository root (avoid relying on Hydra's runtime cwd).
    # File is at: <repo_root>/building2building/baselines/online_trainer.py
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
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    set_random_seed(config.seed)

    # Envs and callbacks
    train_env, eval_env = _make_envs(config, output_dir)
    callbacks = _make_callbacks(config, eval_env, model_dir, log_dir)

    # Build model based on config
    model = build_sb3_model(config, train_env, tb_dir)

    # Train
    model.learn(
        total_timesteps=int(config.training.total_timesteps),
        callback=[callbacks],
    )

    # Test the saved/best policy for a few episodes
    test_cfg = config
    policy_model = load_best_model(config, model_dir) or model
    test_policy(test_cfg, policy_model, output_dir)

    log_test_dir_graphs_wandb(output_dir / "test")

    if wandb_run is not None:
        wandb_run.finish()
