"""Train an SB3 agent on a single building from multizones_reference_buildings.

The building is identified by (building_type, split, index) where *index* is the
0-based position in the pre-generated train/test ID list for that building type.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from algorithms.sb3_utils import build_sb3_model, load_best_model
from b2b.baselines.callbacks import TrainingEpisodeRewardCallback
from b2b.baselines.wandb_utils import (
    finish_wandb_if_started,
    init_wandb_from_config,
)
from b2b.benchmark.rollout_multizones import build_config
from b2b.simulator import create_simulator
from b2b.simulator.wrappers import NormalizeObservation
from b2b.sources import multizones_reference_buildings as mz_source
from b2b.sources.multizones_reference_buildings import (
    BuildingType,
    SPLIT_DATA_DIR,
    search_buildings,
)
from b2b.types import TaskConfig

logger = logging.getLogger(__name__)

def load_split_ids(
    building_type: BuildingType,
    split: Literal["train", "test"],
) -> list[int]:
    return mz_source.load_split_ids(
        building_type=building_type,
        split=split,
        split_data_dir=SPLIT_DATA_DIR,
    )


def _resolve_building_id(
    building_type: BuildingType,
    split: Literal["train", "test"],
    index: int,
) -> int:
    ids = load_split_ids(building_type, split)
    if index < 0 or index >= len(ids):
        raise IndexError(
            f"Index {index} out of range for {building_type}/{split} "
            f"(has {len(ids)} buildings, valid: 0..{len(ids) - 1})"
        )
    return int(ids[index])


def _get_building_row(
    building_type: BuildingType,
    building_id: int,
    *,
    run_period: str,
) -> dict[str, Any]:
    df = search_buildings(
        building_type=building_type,
        building_id=building_id,
        run_period=run_period,
    )
    if df.empty:
        raise ValueError(
            f"No building found: type={building_type}, id={building_id}"
        )
    return dict(df.iloc[0])


def make_multizones_env(
    building_type: BuildingType,
    split: Literal["train", "test"],
    index: int,
    eplus_output_dir: str | Path,
    reward_section: dict[str, Any] | None = None,
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
    energy_weight:
        Weight for the energy term in the reward.
    max_steps:
        If given, wrap the env with ``TimeLimit``.
    """
    task_cfg = TaskConfig.from_dict(task_section or {})
    building_id = _resolve_building_id(building_type, split, index)
    row = _get_building_row(
        building_type,
        building_id,
        run_period=task_cfg.run_period.name,
    )

    out_dir = Path(eplus_output_dir) / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

    bldg_config = build_config(
        row,
        out_dir,
        reward_section=reward_section,
        task_section=task_section,
    )
    env = create_simulator(bldg_config)

    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env


# ------------------------------------------------------------------
# SB3 training loop
# ------------------------------------------------------------------


def _make_single_env(
    building_type: BuildingType,
    split: Literal["train", "test"],
    index: int,
    eplus_root: str,
    reward_section: dict[str, Any],
    task_section: dict[str, Any],
    max_steps: int | None,
    norm_obs: bool,
) -> gym.Env:
    """Create one wrapped env instance (top-level for picklability by SubprocVecEnv)."""
    env = make_multizones_env(
        building_type=building_type,
        split=split,
        index=index,
        eplus_output_dir=eplus_root,
        reward_section=reward_section,
        task_section=task_section,
        max_steps=max_steps,
    )
    env = Monitor(env)
    if norm_obs:
        env = NormalizeObservation(env)
    return env


def _make_envs(
    config: OmegaConf,
    output_dir: Path,
    building_type: BuildingType,
    split: Literal["train", "test"],
    index: int,
    reward_section: dict[str, Any],
    task_section: dict[str, Any],
    max_steps: int | None,
) -> tuple[DummyVecEnv | SubprocVecEnv, DummyVecEnv]:
    norm_obs: bool = config.env.normalize_obs
    common_kwargs = dict(
        building_type=building_type,
        split=split,
        index=index,
        reward_section=reward_section,
        task_section=task_section,
        max_steps=max_steps,
        norm_obs=norm_obs,
    )

    num_envs = int(config.training.num_train_envs)
    train_root = str(output_dir / "train_eplus_outputs")

    if num_envs <= 1:
        train_env: DummyVecEnv | SubprocVecEnv = DummyVecEnv(
            [lambda: _make_single_env(eplus_root=train_root, **common_kwargs)]
        )
    else:
        train_env = SubprocVecEnv(
            [
                lambda r=train_root: _make_single_env(eplus_root=r, **common_kwargs)
                for _ in range(num_envs)
            ]
        )

    eval_root = str(output_dir / "eval_eplus_outputs")
    eval_env = DummyVecEnv(
        [lambda: _make_single_env(eplus_root=eval_root, **common_kwargs)]
    )

    return train_env, eval_env


def _make_callbacks(
    config: OmegaConf,
    eval_env: DummyVecEnv | SubprocVecEnv,
    model_dir: Path,
    log_dir: Path,
) -> CallbackList:
    callbacks: list[Any] = []

    eval_cb = EvalCallback(
        eval_env,
        log_path=str(log_dir),
        eval_freq=config.training.eval_freq,
        best_model_save_path=str(model_dir),
        n_eval_episodes=config.training.eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_cb)
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

    # ---- read multizones selection from config ----
    mz = config.multizones
    building_type: BuildingType = str(mz.building_type)  # type: ignore[assignment]
    split: Literal["train", "test"] = str(mz.split)  # type: ignore[assignment]
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
        else task_cfg.run_period.expected_steps()
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
