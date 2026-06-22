#!/usr/bin/env python3
"""Train per-building PPO specialists (Paper Section 5, Appendix E.2).

Each building type x task x building combination gets its own PPO policy
trained from scratch, then evaluated for one episode.  Results are
collected into a CSV summary.

Usage with Hydra::

    python -m baselines.train_ppo experiment=train_ppo
    python -m baselines.train_ppo experiment=train_ppo \
        building_types=[OfficeSmall] tasks=[task_const_e0] buildings_per_type=4
    python -m baselines.train_ppo experiment=train_ppo \
        building_types=[OfficeSmall] tasks=[task_const_e0] \
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

import building2building as b2b
from baselines.utils.evaluation import run_episode
from baselines.utils.training import build_ppo, make_rl_env_fn, make_vec_env

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of training and evaluating one PPO specialist."""

    building_type: str
    building_id: str
    task: str
    total_reward: float
    normalized_score: float


def _policy_overrides(policy_cfg: DictConfig) -> dict[str, Any]:
    """Extract PPO constructor overrides from the Hydra policy config group."""
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
        building_id for building_id in selected_ids if building_id not in available_set
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
    """Train a PPO specialist on one building and run one eval episode."""
    tag = f"{building_type}/{building_id}/{task}"
    logger.info("Training PPO on %s for %d timesteps", tag, total_timesteps)

    env_fn = make_rl_env_fn(
        building_type=building_type,
        building_id=building_id,
        task=task,
        run_period=run_period,
        normalize_obs=True,
        rescale_action=True,
    )
    env_fns = [env_fn for _ in range(n_envs)]
    vec_env = make_vec_env(env_fns, use_subproc=n_envs > 1)

    model_dir = output_dir / "models" / building_type / task
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_ppo(
        vec_env,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=seed,
        **policy_overrides,
    )
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model_path = model_dir / f"ppo_{building_id}"
    model.save(str(model_path))
    logger.info("Saved model to %s", model_path)
    vec_env.close()

    eval_env = make_rl_env_fn(
        building_type=building_type,
        building_id=building_id,
        task=task,
        run_period=run_period,
        normalize_obs=True,
        rescale_action=True,
        monitor=False,
    )()
    try:
        result = run_episode(eval_env, model)
        total_reward = result.total_reward
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
        eval_env.close()

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
                project=OmegaConf.select(wandb_cfg, "project", default="b2b-baselines"),
                entity=OmegaConf.select(wandb_cfg, "entity", default=None),
                tags=list(OmegaConf.select(wandb_cfg, "tags", default=[])),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"train_ppo_{'_'.join(building_types)}_{'_'.join(tasks)}",
                group="train_ppo",
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
                    _wandb_log(
                        {
                            "eval/total_reward": result.total_reward,
                            "eval/normalized_score": result.normalized_score,
                            "eval/building_type": result.building_type,
                            "eval/building_id": result.building_id,
                            "eval/task": result.task,
                        }
                    )
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
