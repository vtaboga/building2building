"""Train an SB3 agent on a single building from multizones_reference_buildings.

The building is identified by (building_type, split, index) where *index* is the
0-based position in the pre-generated train/test ID list for that building type.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from b2b.baselines.recycling_vec_env import RecyclingSubprocVecEnv
from b2b.training.sb3_utils import build_sb3_model, load_best_model
from b2b.baselines.callbacks import TrainingEpisodeRewardCallback
from b2b.baselines.wandb_utils import (
    finish_wandb_if_started,
    init_wandb_from_config,
)
from b2b.api import make_multizones_env as make_multizones_env_api
from b2b.simulator.wrappers import NormalizeObservation
from b2b.sources import multizones_reference_buildings as mz_source
from b2b.sources.multizones_reference_buildings import (
    BuildingType,
    SPLIT_DATA_DIR,
)
from b2b.types import TaskConfig

logger = logging.getLogger(__name__)

def load_split_ids(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
) -> list[int]:
    return mz_source.load_split_ids(
        building_type=building_type,
        split=split,
        split_data_dir=SPLIT_DATA_DIR,
    )


def _resolve_building_id(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    index: int,
) -> int:
    ids = load_split_ids(building_type, split)
    if index < 0 or index >= len(ids):
        raise IndexError(
            f"Index {index} out of range for {building_type}/{split} "
            f"(has {len(ids)} buildings, valid: 0..{len(ids) - 1})"
        )
    return int(ids[index])


def make_multizones_env(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    index: int,
    eplus_output_dir: str | Path,
    reward_section: dict[str, Any],
    task_section: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    """Create a Gymnasium env for a single multizones building.

    Parameters
    ----------
    building_type:
        One of the six multizones building types.
    split:
        ``"train"`` or ``"test"``.
    index:
        0-based position in the split's ID list.
    eplus_output_dir:
        Root directory for EnergyPlus output files.
    reward_section:
        Reward configuration dict with a ``"reward_type"`` key.
    task_section:
        Optional task configuration dict.
    max_steps:
        If given, wrap the env with ``TimeLimit``.
    """
    return make_multizones_env_api(
        building_type=building_type,
        split=split,
        split_index=index,
        eplus_output_dir=eplus_output_dir,
        task=task_section,
        reward=reward_section,
        max_steps=max_steps,
    )


# ------------------------------------------------------------------
# SB3 training loop
# ------------------------------------------------------------------


def _make_single_env(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    index: int,
    eplus_root: str,
    reward_section: dict[str, Any],
    task_section: dict[str, Any],
    max_steps: int | None,
    norm_obs: bool,
    norm_action: bool = False,
) -> gym.Env:
    """Create one wrapped env instance (top-level for picklability by SubprocVecEnv)."""
    retries = 3
    last_error: RuntimeError | None = None
    for attempt in range(retries):
        try:
            env = make_multizones_env(
                building_type=building_type,
                split=split,
                index=index,
                eplus_output_dir=eplus_root,
                reward_section=reward_section,
                task_section=task_section,
                max_steps=max_steps,
            )
            break
        except RuntimeError as exc:
            if "No multizones building configuration found" not in str(exc):
                raise
            last_error = exc
            if attempt == retries - 1:
                raise
            # Parallel workers can race while populating cached building artifacts.
            time.sleep(0.5 * (attempt + 1))
    else:
        raise RuntimeError("Failed to create multizones environment") from last_error

    if norm_action:
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = Monitor(env)
    if norm_obs:
        env = NormalizeObservation(env)
    return env


def _make_envs(
    config: OmegaConf,
    output_dir: Path,
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    index: int,
    reward_section: dict[str, Any],
    task_section: dict[str, Any],
    max_steps: int | None,
) -> tuple[VecEnv, DummyVecEnv]:
    norm_obs: bool = config.env.normalize_obs
    norm_action: bool = getattr(config.env, "normalize_action", False)
    common_kwargs = dict(
        building_type=building_type,
        split=split,
        index=index,
        reward_section=reward_section,
        task_section=task_section,
        max_steps=max_steps,
        norm_obs=norm_obs,
        norm_action=norm_action,
    )

    num_envs = int(config.training.num_train_envs)
    train_root = output_dir / "train_eplus_outputs"
    train_root.mkdir(parents=True, exist_ok=True)

    if num_envs <= 1:
        train_env: VecEnv = DummyVecEnv(
            [lambda r=str(train_root / "worker_0"): _make_single_env(eplus_root=r, **common_kwargs)]
        )
    else:
        env_fns = [
            lambda r=str(train_root / f"worker_{i}"): _make_single_env(
                eplus_root=r,
                **common_kwargs,
            )
            for i in range(num_envs)
        ]
        recycle_every = int(getattr(config.training, "recycle_every", 0))
        if recycle_every > 0:
            train_env = RecyclingSubprocVecEnv(env_fns, recycle_every=recycle_every)
        else:
            train_env = SubprocVecEnv(env_fns)

    eval_root = output_dir / "eval_eplus_outputs"
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_env = DummyVecEnv(
        [lambda r=str(eval_root / "worker_0"): _make_single_env(eplus_root=r, **common_kwargs)]
    )

    return train_env, eval_env


def _make_callbacks(
    config: OmegaConf,
    eval_env: DummyVecEnv | SubprocVecEnv,
    model_dir: Path,
    log_dir: Path,
) -> CallbackList:
    callbacks: list[Any] = []

    eval_freq = int(config.training.eval_freq)

    eval_cb = EvalCallback(
        eval_env,
        log_path=str(log_dir),
        eval_freq=eval_freq,
        best_model_save_path=str(model_dir),
        n_eval_episodes=config.training.eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_cb)

    checkpoint_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(model_dir),
        name_prefix="checkpoint",
    )
    callbacks.append(checkpoint_cb)

    callbacks.append(TrainingEpisodeRewardCallback())

    try:
        from wandb.integration.sb3 import WandbCallback

        wandb_cb = WandbCallback(
            gradient_save_freq=config.training.cb_gradient_save_freq,
            model_save_path=str(model_dir),
            verbose=2,
        )
        callbacks.append(wandb_cb)
    except Exception:
        logger.info("wandb SB3 callback not available; skipping.")

    return CallbackList(callbacks)


def multizones_trainer(config: OmegaConf, output_dir: Path) -> None:
    """Train an SB3 agent on a single multizones_reference_buildings building."""
    repo_root = Path(__file__).resolve().parents[2]

    # ---- read building selection from config ----
    mz = config.bldg
    building_type: BuildingType = str(mz.building_type)  # type: ignore[assignment]
    split: Literal["train", "test", "test_small"] = str(mz.split)  # type: ignore[assignment]
    index = int(mz.index)

    building_id = _resolve_building_id(building_type, split, index)
    logger.info(
        "Training on %s / %s / index=%d  →  building_id=%d",
        building_type, split, index, building_id,
    )

    reward_raw = OmegaConf.to_container(config.reward, resolve=True)
    reward_section = reward_raw if isinstance(reward_raw, dict) else {}
    task_node = config.get("task", {})
    if OmegaConf.is_config(task_node):
        task_raw = OmegaConf.to_container(task_node, resolve=True)
        task_section = task_raw if isinstance(task_raw, dict) else {}
    elif isinstance(task_node, dict):
        task_section = task_node
    else:
        task_section = {}
    task_cfg = TaskConfig.from_dict(task_section)

    max_steps_raw = getattr(config.env, "max_steps", None)
    max_steps = (
        int(max_steps_raw)
        if max_steps_raw is not None
        else task_cfg.expected_steps()
    )

    # ---- wandb ----
    wandb_run, started_here = init_wandb_from_config(
        config,
        run_dir=output_dir,
        log_code_root=repo_root,
        sync_tensorboard=True,
    )

    # ---- output dirs ----
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    tb_dir = output_dir / "tb"
    for d in (model_dir, log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_random_seed(config.seed)

    # ---- environments ----
    train_env, eval_env = _make_envs(
        config,
        output_dir,
        building_type,
        split,
        index,
        reward_section,
        task_section,
        max_steps,
    )

    # ---- callbacks ----
    callbacks = _make_callbacks(config, eval_env, model_dir, log_dir)

    # ---- build & train ----
    model = build_sb3_model(config, train_env, tb_dir)
    logger.info(
        "Starting training: %s timesteps",
        config.training.total_timesteps,
    )
    model.learn(
        total_timesteps=int(config.training.total_timesteps),
        callback=[callbacks],
    )

    # ---- save final model ----
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    logger.info("Saved final model to %s", final_path)

    best_model = load_best_model(config, model_dir)
    if best_model is not None:
        logger.info("Best model (by eval callback) available at %s/best_model.zip", model_dir)

    finish_wandb_if_started(wandb_run, started_here=started_here)

    logger.info("Training complete. Outputs in %s", output_dir)
