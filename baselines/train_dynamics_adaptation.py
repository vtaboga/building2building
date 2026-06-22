#!/usr/bin/env python3
"""Train PPO on the dynamics adaptation benchmark.

Implements three approaches from the paper:
  1. **Per-building specialist**: One PPO per building (simplest baseline).
  2. **Multi-building baseline**: Single PPO trained across all train buildings.
  3. **Parameterized (building-param augmented)**: Single PPO with building
     parameters appended to observations for adaptation.

Usage with Hydra::

    python -m baselines.train_dynamics_adaptation \
        experiment=train_dynamics_parameterized

    python -m baselines.train_dynamics_adaptation \
        experiment=train_dynamics_specialist difficulty=medium

    python -m baselines.train_dynamics_adaptation \
        experiment=train_dynamics_baseline training.total_timesteps=2_000_000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import building2building as b2b
from baselines.utils.callbacks import TrainingEpisodeRewardCallback
from baselines.utils.training import build_ppo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------


def _make_env(
    building_type: str,
    building_id: str,
    task: str,
) -> gym.Env:
    """Create a single B2B environment."""
    return b2b.make_env(building_type, building_id=building_id, task=task)


def _apply_wrappers(
    env: gym.Env,
    *,
    pad_obs_to: int | None = None,
    normalize_obs: bool = False,
    augment_params: bool = False,
) -> gym.Env:
    """Apply the multi-building wrapper stack.

    Order: PadObservation -> NormalizeObservation -> AugmentWithBuildingParams -> Monitor
    """
    if pad_obs_to is not None:
        env = b2b.PadObservation(env, target_size=pad_obs_to)
    if normalize_obs:
        env = b2b.wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)
    if augment_params:
        env = b2b.AugmentObservationWithBuildingParams(env, allow_defaults=True)
    env = Monitor(env)
    return env


def _make_resampling_env(
    building_type: str,
    building_ids: list[str],
    task: str,
    *,
    pad_obs_to: int | None,
    normalize_obs: bool,
    augment_params: bool,
    wandb_prefix: str = "train",
) -> gym.Env:
    """Create a ResampleBuildingOnResetWrapper environment."""

    def factory(idx: int) -> gym.Env:
        bid = building_ids[idx]
        return _make_env(building_type, bid, task)

    indices = list(range(len(building_ids)))
    env = b2b.ResampleBuildingOnResetWrapper(
        factory, indices, wandb_prefix=wandb_prefix
    )
    return _apply_wrappers(
        env,
        pad_obs_to=pad_obs_to,
        normalize_obs=normalize_obs,
        augment_params=augment_params,
    )


def _detect_max_obs_dim(building_type: str, building_ids: list[str], task: str) -> int:
    """Probe a few environments to find the max observation dimension."""
    max_dim = 0
    sample_ids = building_ids[: min(5, len(building_ids))]
    for bid in sample_ids:
        env = _make_env(building_type, bid, task)
        dim = env.observation_space.shape[0]
        max_dim = max(max_dim, dim)
        env.close()
    return max_dim


# ---------------------------------------------------------------------------
# Training approaches
# ---------------------------------------------------------------------------


def train_specialist(
    building_type: str,
    building_ids: list[str],
    task: str,
    *,
    total_timesteps: int,
    output_dir: Path,
    seed: int,
) -> None:
    """Train one PPO per building (per-building specialist)."""
    for i, bid in enumerate(building_ids):
        logger.info(
            "Training specialist %d/%d on %s/%s",
            i + 1,
            len(building_ids),
            building_type,
            bid,
        )

        def make_fn() -> gym.Env:
            env = _make_env(building_type, bid, task)
            return Monitor(env)

        vec_env = DummyVecEnv([make_fn])
        model = build_ppo(
            vec_env,
            tensorboard_log=str(output_dir / "tb" / bid),
            seed=seed + i,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=[TrainingEpisodeRewardCallback()],
        )
        model.save(str(output_dir / "models" / f"specialist_{bid}"))
        vec_env.close()


def train_multi_building(
    building_type: str,
    train_ids: list[str],
    task: str,
    *,
    total_timesteps: int,
    n_envs: int,
    output_dir: Path,
    seed: int,
    augment_params: bool = False,
) -> None:
    """Train a single PPO across many buildings."""
    pad_obs_to = _detect_max_obs_dim(building_type, train_ids, task)
    logger.info("Padding observations to %d", pad_obs_to)
    (output_dir / "metadata.json").write_text(json.dumps({"pad_obs_size": pad_obs_to}))

    def make_train_fn(idx: int) -> Callable[[], gym.Env]:
        def fn() -> gym.Env:
            return _make_resampling_env(
                building_type,
                train_ids,
                task,
                pad_obs_to=pad_obs_to,
                normalize_obs=True,
                augment_params=augment_params,
                wandb_prefix=f"train_{idx}",
            )

        return fn

    env_fns = [make_train_fn(i) for i in range(n_envs)]
    if n_envs > 1:
        raw_env = SubprocVecEnv(env_fns)
    else:
        raw_env = DummyVecEnv(env_fns)

    train_env = VecNormalize(raw_env, norm_obs=False, norm_reward=True)

    model = build_ppo(
        train_env,
        tensorboard_log=str(output_dir / "tb"),
        seed=seed,
    )

    approach = "parameterized" if augment_params else "baseline"
    logger.info(
        "Training %s multi-building PPO (%d envs, %d buildings)",
        approach,
        n_envs,
        len(train_ids),
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[TrainingEpisodeRewardCallback()],
        progress_bar=True,
    )

    model.save(str(output_dir / "models" / f"multi_{approach}"))
    train_env.close()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    difficulty: str = cfg.difficulty
    approach: str = cfg.approach
    task: str = cfg.reward.task_name
    total_timesteps: int = int(cfg.training.total_timesteps)
    n_envs: int = int(cfg.training.n_envs)
    seed: int = int(cfg.seed)
    max_buildings: int | None = cfg.get("max_buildings", None)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    bench = b2b.benchmarks.DynamicsAdaptation(difficulty=difficulty, task=task)
    building_type = bench.building_type
    train_ids = bench.train_building_ids()
    test_ids = bench.test_building_ids()

    if max_buildings is not None:
        train_ids = train_ids[:max_buildings]

    logger.info(
        "Benchmark: %s difficulty=%s  train=%d  test=%d",
        building_type,
        difficulty,
        len(train_ids),
        len(test_ids),
    )
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    wandb_cfg = cfg.get("wandb", {})
    if OmegaConf.select(wandb_cfg, "enabled", default=False):
        try:
            import wandb

            wandb.init(
                project=OmegaConf.select(wandb_cfg, "project", default="b2b-dynamics"),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"{difficulty}_{approach}",
            )
        except ImportError:
            logger.warning("wandb not installed; skipping init")

    if approach == "specialist":
        train_specialist(
            building_type,
            train_ids,
            task,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            seed=seed,
        )
    elif approach in ("baseline", "parameterized"):
        train_multi_building(
            building_type,
            train_ids,
            task,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            output_dir=output_dir,
            seed=seed,
            augment_params=(approach == "parameterized"),
        )
    else:
        raise ValueError(f"Unknown approach: {approach!r}")

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
