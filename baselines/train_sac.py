#!/usr/bin/env python3
"""Train per-building SAC specialists (companion to train_ppo.py).

Each building type x task x building combination gets its own SAC policy
trained from scratch, then evaluated for one episode.  Results are
collected into a CSV summary.

SAC-specific notes:
- Observations are normalised via SB3 VecNormalize (norm_obs=True,
  norm_reward=False).  The VecNormalize state is saved alongside the model
  so deterministic evaluation uses the same normalisation statistics.
- n_envs=4 (vs PPO's 14) because SAC does one gradient step per env step;
  more envs would reduce update density without proportional benefit.
- total_timesteps=1M (vs PPO's 5M) because SAC is off-policy and
  sample-efficient.

Usage with Hydra::

    python -m baselines.train_sac experiment=train_sac
    python -m baselines.train_sac experiment=train_sac \
        building_types=[OfficeSmall] tasks=[task1] buildings_per_type=4
    python -m baselines.train_sac experiment=train_sac \
        building_types=[OfficeSmall] tasks=[task1] \
        building_ids=[OfficeSmall-0001]
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import building2building as b2b
from baselines.utils.evaluation import run_episode
from baselines.utils.training import build_sac, make_vec_env

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of training and evaluating one SAC specialist."""

    building_type: str
    building_id: str
    task: str
    total_reward: float
    normalized_score: float


def _policy_overrides(policy_cfg: DictConfig) -> dict[str, Any]:
    """Extract SAC constructor overrides from the Hydra policy config group."""
    raw: dict[str, Any] = OmegaConf.to_container(policy_cfg, resolve=True)  # type: ignore[assignment]
    for key in ("algorithm", "policy_type", "device"):
        raw.pop(key, None)
    return raw


def _selected_building_ids(
    cfg: DictConfig,
    *,
    building_type: str,
    split: str,
    buildings_per_type: int,
) -> list[str]:
    """Resolve explicit building overrides or split-based defaults."""
    explicit_ids_raw = cfg.get("building_ids")
    available_ids = list(b2b.list_buildings(building_type, split=split))
    available_set = set(available_ids)
    if explicit_ids_raw is None:
        if buildings_per_type <= 0:
            return available_ids
        return available_ids[:buildings_per_type]

    explicit_ids = [
        str(building_id)
        for building_id in OmegaConf.to_container(explicit_ids_raw, resolve=True)
    ]
    selected_ids = [
        building_id
        for building_id in explicit_ids
        if building_id.startswith(f"{building_type}-")
    ]
    invalid_ids = [
        building_id
        for building_id in selected_ids
        if building_id not in available_set
    ]
    if invalid_ids:
        raise ValueError(
            f"Invalid building_ids for {building_type}/{split}: {invalid_ids}. "
            "Ensure IDs belong to the selected split."
        )
    return selected_ids


def train_and_eval(
    building_type: str,
    building_id: str,
    task: str,
    *,
    policy_overrides: dict[str, Any],
    total_timesteps: int,
    n_envs: int,
    output_dir: Path,
    seed: int,
    run_period: str = "full_year",
) -> TrainResult:
    """Train a SAC specialist on one building and run one eval episode."""
    tag = f"{building_type}/{building_id}/{task}"
    logger.info("Training SAC on %s for %d timesteps", tag, total_timesteps)

    def make_env() -> Monitor:
        env = b2b.new_make_env(
            building_type,
            building_id=building_id,
            task=task,
            run_period=run_period,
        )
        return Monitor(env)

    env_fns = [make_env for _ in range(n_envs)]
    raw_vec_env = make_vec_env(env_fns, use_subproc=n_envs > 1)
    vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=False)

    model_dir = output_dir / "models" / building_type / task
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_sac(
        vec_env,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=seed,
        **policy_overrides,
    )
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model_path = model_dir / f"sac_{building_id}"
    model.save(str(model_path))
    vec_normalize_path = model_dir / f"sac_vecnormalize_{building_id}.pkl"
    vec_env.save(str(vec_normalize_path))
    logger.info("Saved model to %s", model_path)
    logger.info("Saved VecNormalize state to %s", vec_normalize_path)
    vec_env.close()

    # Eval: wrap a fresh single env in DummyVecEnv, then load saved VecNormalize.
    eval_inner = b2b.new_make_env(
        building_type,
        building_id=building_id,
        task=task,
        run_period=run_period,
    )
    eval_vec = DummyVecEnv([lambda: eval_inner])  # type: ignore[return-value]
    eval_vec_norm = VecNormalize.load(str(vec_normalize_path), eval_vec)
    eval_vec_norm.training = False
    eval_vec_norm.norm_reward = False

    try:
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, _ = evaluate_policy(
            model,
            eval_vec_norm,
            n_eval_episodes=1,
            deterministic=True,
        )
        total_reward = float(mean_reward)
        logger.info("Eval reward for %s: %.1f", tag, total_reward)

        normalized_score = b2b.compute_normalized_score(
            total_reward,
            building_type,
            task,
            run_period=run_period,
            building_id=building_id,
        )
        logger.info("Normalized score for %s: %.4f", tag, normalized_score)
    finally:
        eval_vec_norm.close()

    return TrainResult(
        building_type=building_type,
        building_id=building_id,
        task=task,
        total_reward=total_reward,
        normalized_score=normalized_score,
    )


def write_results_csv(results: list[TrainResult], path: Path) -> None:
    """Write training results to a CSV summary file."""
    fieldnames = [
        "building_type",
        "building_id",
        "task",
        "total_reward",
        "normalized_score",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(
            results, key=lambda x: (x.building_type, x.task, x.building_id)
        ):
            writer.writerow(
                {
                    "building_type": r.building_type,
                    "building_id": r.building_id,
                    "task": r.task,
                    "total_reward": f"{r.total_reward:.1f}",
                    "normalized_score": f"{r.normalized_score:.4f}",
                }
            )
    logger.info("Wrote %d results to %s", len(results), path)


def _wandb_log(payload: dict[str, Any]) -> None:
    """Log to wandb if a run is active; silently no-op otherwise."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        pass


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    building_types: list[str] = list(cfg.building_types)
    tasks: list[str] = list(cfg.tasks)
    split: str = cfg.get("split", "test")
    buildings_per_type: int = int(cfg.buildings_per_type)
    total_timesteps: int = int(cfg.training.total_timesteps)
    n_envs: int = int(cfg.training.n_envs)
    seed: int = int(cfg.get("seed", 0))
    run_period: str = str(cfg.get("run_period", "full_year"))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(OmegaConf.select(wandb_cfg, "enabled", default=False))
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=OmegaConf.select(
                    wandb_cfg, "project", default="b2b-baselines"
                ),
                entity=OmegaConf.select(wandb_cfg, "entity", default=None),
                tags=list(OmegaConf.select(wandb_cfg, "tags", default=[])),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"sac_{'_'.join(building_types)}_{'_'.join(tasks)}",
                group="train_sac",
                sync_tensorboard=True,
            )
        except ImportError:
            logger.warning("wandb not installed; skipping init")
            use_wandb = False

    overrides = _policy_overrides(cfg.policy)
    results: list[TrainResult] = []

    for bt in building_types:
        building_ids = _selected_building_ids(
            cfg,
            building_type=bt,
            split=split,
            buildings_per_type=buildings_per_type,
        )
        logger.info(
            "Building type %s: %d buildings x %d tasks",
            bt,
            len(building_ids),
            len(tasks),
        )

        for task in tasks:
            for bid in building_ids:
                try:
                    result = train_and_eval(
                        bt,
                        bid,
                        task,
                        policy_overrides=overrides,
                        total_timesteps=total_timesteps,
                        n_envs=n_envs,
                        output_dir=output_dir,
                        seed=seed,
                        run_period=run_period,
                    )
                    results.append(result)
                    _wandb_log({
                        "eval/total_reward": result.total_reward,
                        "eval/normalized_score": result.normalized_score,
                        "eval/building_type": result.building_type,
                        "eval/building_id": result.building_id,
                        "eval/task": result.task,
                    })
                except Exception:
                    logger.exception("Failed: %s/%s task=%s", bt, bid, task)

    if results:
        write_results_csv(results, output_dir / "results.csv")
    else:
        logger.warning("No results to write.")

    if use_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
