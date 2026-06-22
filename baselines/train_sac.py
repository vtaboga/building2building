#!/usr/bin/env python3
"""Train per-building SAC specialists (companion to train_ppo.py).

Each building type x task x building combination gets its own SAC policy
trained from scratch, then evaluated for one episode.  Results are
collected into a CSV summary.

SAC-specific notes:
- Observations are normalised via deterministic per-feature ``[0, 1]``
  scaling (``b2b.wrap_env_for_rl``).  No ``VecNormalize`` statistics file
  is written; evaluation is reproducible by construction.
- Actions are rescaled to ``[-1, 1]`` via ``gym.wrappers.RescaleAction``
  inside the env (applied before ``TimeLimit``); ``wrap_env_for_rl`` is
  called with ``rescale_action=False`` to avoid a second rescale layer.
- ``train_freq=1, gradient_steps=-1`` → SB3 performs ``n_envs`` gradient
  steps per environment step, keeping the update-to-data ratio at 1.0.
- Tasks use ``NormalizedDeadbandReward`` (``task_*_e0`` family by default)
  with per-bucket ``(τ_T, τ_E)`` constants from
  ``reward_normalizers.yaml``.  ``energy_weight=0`` (``e0``)
  means the reward measures pure thermal comfort; switch to ``emed``/
  ``ehigh`` to add an energy penalty.
- total_timesteps defaults to 2M in experiment=train_sac (vs PPO's 5M)
  because SAC is off-policy and sample-efficient.
- Multi-seed runs: use Hydra multirun, e.g.
      python -m baselines.train_sac experiment=train_sac_task_study \\
          --multirun seed=0,1,2

Usage with Hydra::

    python -m baselines.train_sac experiment=train_sac
    python -m baselines.train_sac experiment=train_sac \\
        building_types=[OfficeSmall] tasks=[task_const_e0] buildings_per_type=1
    python -m baselines.train_sac experiment=train_sac \\
        building_types=[OfficeSmall] tasks=[task_const_e0] \\
        building_ids=[OfficeSmall-0001]

Quick smoke-test (single small-office building, 50 k steps, 1 env)::

    python -m baselines.train_sac experiment=train_sac \\
        building_types=[OfficeSmall] \\
        tasks=[task_const_e0] \\
        buildings_per_type=1 \\
        training.n_envs=1 \\
        training.total_timesteps=50000
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback

import building2building as b2b
from baselines.utils.evaluation import run_episode
from baselines.utils.training import build_sac, make_rl_env_fn, make_vec_env

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
    checkpoint_freq: int = 200_000,
) -> TrainResult:
    """Train a SAC specialist on one building and run one eval episode.

    A :class:`CheckpointCallback` saves the model roughly every
    ``checkpoint_freq`` environment steps so that a job killed by the SLURM
    wall-clock limit still leaves a usable partial policy on disk (SAC
    otherwise only writes the model after ``learn`` returns).
    """
    tag = f"{building_type}/{building_id}/{task}"
    logger.info("Training SAC on %s for %d timesteps", tag, total_timesteps)

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

    model = build_sac(
        vec_env,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=seed,
        **policy_overrides,
    )
    # SB3 seeds the policy, replay buffer RNG, and each VecEnv worker
    # (worker i gets seed+i) via set_random_seed, called internally by
    # the SAC constructor.  Calling it again here is a no-op but makes
    # the intent explicit and guards against future constructor changes.
    model.set_random_seed(seed)

    # CheckpointCallback counts its own ``_on_step`` calls, which fire once
    # per VecEnv step (i.e. every ``n_envs`` environment steps), so divide
    # the desired step interval by ``n_envs``.  Saved as
    # ``models/<bt>/<task>/checkpoints/sac_<id>_<steps>_steps.zip``; the
    # replay buffer is not saved (it is large and only needed to *resume*,
    # not to evaluate the partial policy).
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix=f"sac_{building_id}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
    )

    model_path = model_dir / f"sac_{building_id}"
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

        try:
            normalized_score = b2b.compute_normalized_score(
                total_reward,
                building_type,
                task,
                run_period=run_period,
                building_id=building_id,
            )
            logger.info("Normalized score for %s: %.4f", tag, normalized_score)
        except KeyError:
            logger.warning(
                "No baseline return found for %s — normalized_score set to nan. "
                "Run baselines/scripts/run_baseline_returns.sh for task=%s then "
                "merge_baseline_returns.py to populate baseline_returns.csv.",
                tag,
                task,
            )
            normalized_score = float("nan")
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
            norm_str = (
                "nan" if math.isnan(r.normalized_score) else f"{r.normalized_score:.4f}"
            )
            writer.writerow(
                {
                    "building_type": r.building_type,
                    "building_id": r.building_id,
                    "task": r.task,
                    "total_reward": f"{r.total_reward:.1f}",
                    "normalized_score": norm_str,
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
    checkpoint_freq: int = int(cfg.training.get("checkpoint_freq", 200_000))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(OmegaConf.select(wandb_cfg, "enabled", default=False))
    if use_wandb:
        try:
            import wandb

            # Build a unique run name encoding the full training context so
            # that array jobs (one building per job) are distinguishable in
            # the W&B UI.  Format: sac_<BT>_<task>_<building_id>_s<seed>
            # When multiple BTs or tasks are in one job, join with "+".
            bt_str = "+".join(building_types)
            task_str = "+".join(tasks)
            # building_ids is not resolved yet here; use cfg value if present.
            explicit_ids = cfg.get("building_ids")
            if explicit_ids is not None:
                ids_list = list(OmegaConf.to_container(explicit_ids, resolve=True))
                id_str = "+".join(str(i) for i in ids_list)
            else:
                id_str = f"{buildings_per_type}bldgs"
            run_name = f"sac_{bt_str}_{task_str}_{id_str}_s{seed}"

            wandb.init(
                project=OmegaConf.select(wandb_cfg, "project", default="b2b-baselines"),
                entity=OmegaConf.select(wandb_cfg, "entity", default=None),
                tags=list(OmegaConf.select(wandb_cfg, "tags", default=[])),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=run_name,
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
                        checkpoint_freq=checkpoint_freq,
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
