#!/usr/bin/env python3
"""Tune PPO hyperparameters with the CHS procedure (Patterson et al., RLC 2024).

A single Orion experiment evaluates each HP config across buildings sampled from
all building types and ASHRAE climate zones simultaneously.  Each building is one
environment: the agent trains and evaluates on the same building, matching the
paper's setup.  This gives the CHS score direct cross-type and cross-CZ signal
without a separate post-hoc aggregation step.

**Sweep mode** (default) -- run as a SLURM job array worker::

    python -m baselines.tune_ppo experiment=tune_ppo task=task1

**CHS analysis** (after all sweep jobs complete)::

    python -m baselines.tune_ppo experiment=tune_ppo task=task1 analyze=true

**Re-evaluation** (train+eval champion config with many seeds)::

    python -m baselines.tune_ppo experiment=tune_ppo task=task1 reeval=true
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from orion.client import create_experiment
from orion.core.utils.exceptions import ReservationRaceCondition
import building2building as b2b
from baselines.chs import load_trial_rewards_from_dir
from baselines.utils.evaluation import run_episode
from baselines.utils.training import build_ppo, make_rl_env_fn, make_vec_env

logger = logging.getLogger(__name__)


# batch_size upper bound (8192) exceeds the smallest rollout buffer
# (n_steps=512 × n_envs=8 = 4096), so the clamp in _params_to_ppo_hparams
# now fires for that combination.
ORION_SPACE: dict[str, str] = {
    "/learning_rate": "loguniform(1e-5, 5e-4)",
    "/n_steps": "choices([512, 1024, 2048])",
    "/batch_size": "choices([256, 512, 1024, 2048, 4096, 8192])",
    "/ent_coef": "loguniform(5e-4, 5e-2)",
    "/gamma": "choices([0.97, 0.98, 0.99, 0.995])",
    "/n_epochs": "choices([5, 10, 20])",
}

# PPO parameters not in ORION_SPACE — fixed as robust across tasks.
_FIXED_PPO_HPARAMS: dict[str, Any] = {
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
}

_FIXED_POLICY_KWARGS: dict[str, Any] = {
    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    "activation_fn": "Tanh",
    "ortho_init": True,
    "log_std_init": -1.0,
}

BUILDING_TYPES: list[str] = [
    "OfficeSmall",
    "OfficeMedium",
    "RestaurantFastFood",
    "RetailStandalone",
    "Warehouse",
]

# Recorded at process startup so _run_sweep can compute elapsed wall time.
_PROCESS_START: float = time.time()

# Conservative per-(building × seed) training time used to estimate trial cost
# before calling experiment.suggest().  Overestimating is safe; underestimating
# risks leaving a reserved trial stuck in Orion if the worker is killed.
_SECS_PER_BUILDING_SEED: int = 90 * 60  # 90 min (generous vs. ~75 min observed)


# ── Climate zone helpers ─────────────────────────────────────────────


def _get_climate_zone(building_type: str, building_id: str) -> int | None:
    """Return the ASHRAE climate zone of a building, or ``None`` if it has
    no CZ assignment (e.g. :class:`SingleFamilyHouse`)."""
    if building_type in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
        return None
    return b2b.get_climate_zone(building_type, building_id)


def _group_by_climate_zone(
    building_type: str, building_ids: list[str]
) -> dict[int, list[str]]:
    """Group building IDs by climate zone.

    Buildings whose CZ cannot be determined are placed in group ``-1``.
    """
    groups: dict[int, list[str]] = defaultdict(list)
    for bid in building_ids:
        cz = _get_climate_zone(building_type, bid)
        groups[cz if cz is not None else -1].append(bid)
    return dict(groups)


def _sample_one_per_cz(
    building_type: str,
    split: str,
    rng: random.Random,
) -> list[str]:
    """Sample one building per climate zone (LHS stratification).

    For building types without CZ info (e.g. SingleFamilyHouse), fall
    back to sampling 8 buildings randomly.
    """
    all_ids = b2b.list_buildings(building_type, split=split)
    if not all_ids:
        return []
    groups = _group_by_climate_zone(building_type, all_ids)

    if list(groups.keys()) == [-1]:
        n = min(8, len(all_ids))
        return rng.sample(all_ids, n)

    sampled: list[str] = []
    for cz in sorted(groups):
        if cz == -1:
            continue
        sampled.append(rng.choice(groups[cz]))
    return sampled


# ── Orion params to PPO hparams ─────────────────────────────────────


def _params_to_ppo_hparams(params: dict[str, Any], n_envs: int = 1) -> dict[str, Any]:
    """Convert flat Orion trial params to PPO constructor kwargs.

    ``batch_size`` is clamped to ``n_steps * n_envs`` so SB3 never receives
    a batch larger than the rollout buffer.  With batch_size up to 8192 in
    the search space this fires for e.g. batch_size=8192 with n_steps=512
    (rollout buffer = 4096).  All choices are powers of two, so the clamped
    value always divides the rollout buffer exactly.
    """
    n_steps = int(params["/n_steps"])
    batch_size = min(int(params["/batch_size"]), n_steps * n_envs)
    return {
        "learning_rate": params["/learning_rate"],
        "n_steps": n_steps,
        "batch_size": batch_size,
        "ent_coef": params["/ent_coef"],
        "gamma": float(params["/gamma"]),
        "n_epochs": int(params["/n_epochs"]),
        **_FIXED_PPO_HPARAMS,
        "policy_kwargs": dict(_FIXED_POLICY_KWARGS),
    }


# ── Single train-and-eval ───────────────────────────────────────────


def _wandb_log(payload: dict[str, Any]) -> None:
    """Log to wandb if a run is active; silently no-op otherwise."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        pass


def _train_and_eval_single(
    building_type: str,
    train_building_id: str,
    eval_building_id: str,
    task: str,
    hparams: dict[str, Any],
    total_timesteps: int,
    n_envs: int,
    seed: int,
    tensorboard_log: str | None = None,
    verbose: int = 1,
) -> float:
    """Train PPO on *train_building_id*, evaluate on *eval_building_id*.

    Returns the total episode reward on the eval building.
    """

    env_fn = make_rl_env_fn(
        building_type=building_type,
        building_id=train_building_id,
        task=task,
        normalize_obs=True,
        rescale_action=True,
    )
    env_fns = [env_fn for _ in range(n_envs)]
    vec_env = make_vec_env(env_fns, use_subproc=n_envs > 1)

    try:
        model = build_ppo(
            vec_env,
            seed=seed,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **hparams,
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
    finally:
        vec_env.close()

    eval_env = make_rl_env_fn(
        building_type=building_type,
        building_id=eval_building_id,
        task=task,
        normalize_obs=True,
        rescale_action=True,
        monitor=False,
    )()
    try:
        result = run_episode(eval_env, model)
        return result.total_reward
    finally:
        eval_env.close()


# ── Per-trial reward I/O ─────────────────────────────────────────────


def _save_trial_rewards(
    trial_id: str,
    trial_idx: int,
    rewards: dict[str, list[float]],
    results_dir: Path,
    params: dict[str, Any] | None = None,
) -> None:
    """Write per-building rewards for one trial to a JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": trial_id,
        "trial_idx": trial_idx,
        "params": params or {},
        "rewards": rewards,
    }
    out_path = results_dir / f"{trial_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))


# ── HP config I/O ────────────────────────────────────────────────────


def _trial_params_to_ppo_config(params: dict[str, Any]) -> dict[str, Any]:
    """Convert flat Orion trial params to a nested PPO config dict for YAML.

    The saved batch_size is the raw Orion-suggested value; the runtime clamp
    in _params_to_ppo_hparams applies when training with a specific n_envs.
    """
    return {
        "algorithm": "ppo",
        "policy_type": "MlpPolicy",
        "device": "auto",
        "learning_rate": float(params["/learning_rate"]),
        "n_steps": int(params["/n_steps"]),
        "batch_size": int(params["/batch_size"]),
        "ent_coef": float(params["/ent_coef"]),
        "gamma": float(params["/gamma"]),
        "n_epochs": int(params["/n_epochs"]),
        **{
            k: float(v) if isinstance(v, float) else v
            for k, v in _FIXED_PPO_HPARAMS.items()
        },
        "policy_kwargs": dict(_FIXED_POLICY_KWARGS),
    }


def _save_best_config(
    params: dict[str, Any],
    output_dir: Path,
    task: str,
) -> Path:
    """Save the champion HP config as a YAML file."""
    cfg = _trial_params_to_ppo_config(params)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ppo_chs_all_{task}.yaml"
    out_path = output_dir / fname
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    logger.info("Saved CHS-tuned PPO config to %s", out_path)
    return out_path


# ── Orion worker loop ────────────────────────────────────────────────


def _run_sweep(
    experiment: Any,
    task: str,
    building_instances: list[tuple[str, str]],
    total_timesteps: int,
    n_envs: int,
    ntune_seeds: int,
    results_dir: Path,
    wall_time_seconds: float = float("inf"),
) -> None:
    """Orion worker loop: suggest trials, train, observe, repeat.

    Each trial evaluates the candidate HP config on every
    ``(building_type, building_id)`` pair in *building_instances*, training
    and evaluating on the same building (one building = one CHS environment).
    TensorBoard logging is disabled during the sweep to avoid filling disk
    quota across many short training runs.

    ``wall_time_seconds`` should match the SLURM ``--time`` limit.  Before
    each ``suggest()`` call the worker checks whether enough wall time remains
    to complete a full trial; if not, it exits cleanly without reserving a
    trial slot.  This prevents stale ``"reserved"`` entries in the Orion DB
    that would otherwise permanently reduce the effective trial budget.
    """
    # Pre-compute the expected trial cost with a safety margin so workers that
    # start late (due to queue delays) exit before they run out of time.
    trial_cost_sec = len(building_instances) * ntune_seeds * _SECS_PER_BUILDING_SEED

    trial_counter = 0
    while not experiment.is_done:
        elapsed = time.time() - _PROCESS_START
        remaining = wall_time_seconds - elapsed
        if remaining < trial_cost_sec:
            logger.info(
                "Wall-time guard: %.1f h remaining < estimated trial cost %.1f h. "
                "Exiting without reserving a new trial.",
                remaining / 3600,
                trial_cost_sec / 3600,
            )
            break

        # Jitter before suggest to reduce simultaneous lock contention on
        # pickleddb when all workers start at the same time (job array launch).
        time.sleep(random.uniform(0, 30))
        trial = None
        for _attempt in range(10):
            try:
                trial = experiment.suggest()
                break
            except ReservationRaceCondition:
                backoff = random.uniform(5, 30)
                logger.warning(
                    "ReservationRaceCondition on suggest() attempt %d; "
                    "retrying in %.0f s",
                    _attempt + 1,
                    backoff,
                )
                time.sleep(backoff)
        if trial is None:
            logger.info("No more trials to suggest; worker exiting.")
            break

        params = trial.params
        hparams = _params_to_ppo_hparams(params, n_envs=n_envs)
        trial_idx = trial_counter
        trial_counter += 1

        logger.info("Worker starting trial %s (idx %d)", trial.id, trial_idx)

        all_rewards: list[float] = []
        rewards_per_building: dict[str, list[float]] = {}
        failed = False
        aborted = False
        total_seeds = len(building_instances) * ntune_seeds
        seeds_done = 0
        trial_start = time.time()

        for btype, bid in building_instances:
            rewards_per_building[bid] = []
            for seed in range(ntune_seeds):
                try:
                    reward = _train_and_eval_single(
                        btype,
                        bid,
                        bid,
                        task,
                        hparams,
                        total_timesteps,
                        n_envs,
                        seed=seed,
                        tensorboard_log=None,
                        verbose=0,
                    )
                except Exception as exc:
                    logger.warning(
                        "Trial %s seed %d failed on %s/%s: %s",
                        trial.id,
                        seed,
                        btype,
                        bid,
                        exc,
                    )
                    failed = True
                    break

                rewards_per_building[bid].append(reward)
                all_rewards.append(reward)
                seeds_done += 1
                logger.info(
                    "Trial %s | %s/%s | seed %d | reward %.1f",
                    trial.id,
                    btype,
                    bid,
                    seed,
                    reward,
                )

                # After each seed, check whether this node is fast enough to
                # finish before the wall-time limit.  A 2× safety margin
                # catches slow nodes (cn-f, ~3× slower) after the first seed
                # without prematurely aborting fast nodes.
                elapsed_trial = time.time() - trial_start
                secs_per_seed = elapsed_trial / seeds_done
                seeds_left = total_seeds - seeds_done
                estimated_remaining = secs_per_seed * seeds_left
                wall_remaining = wall_time_seconds - (time.time() - _PROCESS_START)
                if estimated_remaining > wall_remaining * 0.9:
                    logger.warning(
                        "Pace check: %.1f h estimated to finish trial "
                        "but only %.1f h of wall time remains. "
                        "Aborting trial to avoid SLURM timeout.",
                        estimated_remaining / 3600,
                        wall_remaining / 3600,
                    )
                    experiment._experiment.set_trial_status(trial, "interrupted")
                    aborted = True
                    break

            if failed or aborted:
                break

        if aborted:
            logger.info(
                "Trial %s interrupted (slow node); will be retried by another worker.",
                trial.id,
            )
            break  # this node is too slow for any trial; stop the worker loop

        if failed or len(all_rewards) == 0:
            objective_value = 1e10
        else:
            objective_value = -float(np.mean(all_rewards))

        experiment.observe(
            trial,
            [{"name": "objective", "type": "objective", "value": objective_value}],
        )

        _save_trial_rewards(
            trial.id,
            trial_idx,
            rewards_per_building,
            results_dir,
            params=params,
        )
        logger.info("Trial %s observed (objective=%.1f)", trial.id, objective_value)

        _wandb_log(
            {
                "trial/index": trial_idx,
                "trial/mean_reward": (
                    -objective_value if objective_value < 1e9 else float("nan")
                ),
                "trial/objective": objective_value,
                "trial/failed": failed,
                **{
                    f"trial/reward_{bid}": (
                        float(np.mean(rews)) if rews else float("nan")
                    )
                    for bid, rews in rewards_per_building.items()
                },
            }
        )


# ── CHS analysis ────────────────────────────────────────────────────


def _run_analyze(
    experiment: Any,
    results_dir: Path,
    output_dir: Path,
    task: str,
) -> None:
    """Post-hoc CHS analysis: CDF-normalize and select the best trial."""
    store = load_trial_rewards_from_dir(results_dir)
    if not store.env_ids():
        logger.error("No trial reward files found in %s", results_dir)
        return

    trial_ids = store.trial_ids()
    logger.info(
        "Loaded rewards for %d trials across %d buildings",
        len(trial_ids),
        len(store.env_ids()),
    )

    ranking = store.trial_summary(trial_ids)
    logger.info("CHS ranking (top 5):")
    for rank, (tid, cdf_score) in enumerate(ranking[:5], 1):
        logger.info("  #%d  trial %s  CDF score %.4f", rank, tid, cdf_score)

    chs_best_id = ranking[0][0]

    best_params: dict[str, Any] | None = None
    for trial in experiment.fetch_trials():
        if trial.id == chs_best_id:
            best_params = trial.params
            break

    if best_params is None:
        logger.error("Could not find trial %s in Orion experiment.", chs_best_id)
        return

    _save_best_config(best_params, output_dir / "configs", task)
    logger.info("CHS analysis complete. Best trial: %s", chs_best_id)


# ── Re-evaluation ────────────────────────────────────────────────────


def _run_reeval(
    experiment: Any,
    task: str,
    building_instances: list[tuple[str, str]],
    total_timesteps: int,
    n_envs: int,
    reeval_seeds: int,
    results_dir: Path,
    output_dir: Path,
    tensorboard_log: str | None = None,
) -> None:
    """Re-evaluate the CHS-best trial with many seeds on all buildings."""
    store = load_trial_rewards_from_dir(results_dir)
    trial_ids = store.trial_ids()
    if not trial_ids:
        logger.error("No trial reward files found; run the sweep first.")
        return

    ranking = store.trial_summary(trial_ids)
    chs_best_id = ranking[0][0]

    best_params: dict[str, Any] | None = None
    for trial in experiment.fetch_trials():
        if trial.id == chs_best_id:
            best_params = trial.params
            break

    if best_params is None:
        logger.error("Could not find trial %s in Orion experiment.", chs_best_id)
        return

    hparams = _params_to_ppo_hparams(best_params, n_envs=n_envs)
    logger.info(
        "Re-evaluating CHS-best trial %s on %d buildings x %d seeds",
        chs_best_id,
        len(building_instances),
        reeval_seeds,
    )

    results: dict[str, list[float]] = {}
    for btype, bid in building_instances:
        results[bid] = []
        for seed in range(reeval_seeds):
            try:
                reward = _train_and_eval_single(
                    btype,
                    bid,
                    bid,
                    task,
                    hparams,
                    total_timesteps,
                    n_envs,
                    seed=seed,
                    tensorboard_log=tensorboard_log,
                )
                results[bid].append(reward)
                logger.info(
                    "Reeval %s/%s seed %d: reward %.1f",
                    btype,
                    bid,
                    seed,
                    reward,
                )
                _wandb_log(
                    {
                        "reeval/building": bid,
                        "reeval/building_type": btype,
                        "reeval/seed": seed,
                        "reeval/reward": reward,
                    }
                )
            except Exception:
                logger.exception("Reeval failed on %s/%s seed %d", btype, bid, seed)

    reeval_dir = output_dir / "reeval"
    reeval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reeval_dir / f"reeval_all_{task}.yaml"

    summary: dict[str, Any] = {
        "task": task,
        "best_trial": chs_best_id,
        "reeval_seeds": reeval_seeds,
        "total_timesteps": total_timesteps,
        "buildings": {},
    }
    all_rewards: list[float] = []
    for btype, bid in building_instances:
        rewards = results[bid]
        arr = np.array(rewards)
        all_rewards.extend(rewards)
        summary["buildings"][bid] = {
            "building_type": btype,
            "mean": float(arr.mean()) if len(arr) > 0 else None,
            "std": float(arr.std()) if len(arr) > 0 else None,
            "n": len(rewards),
            "rewards": [float(r) for r in rewards],
        }

    all_arr = np.array(all_rewards)
    summary["overall_mean"] = float(all_arr.mean()) if len(all_arr) > 0 else None
    summary["overall_std"] = float(all_arr.std()) if len(all_arr) > 0 else None

    summary_path.write_text(
        yaml.dump(summary, default_flow_style=False, sort_keys=False)
    )
    logger.info("Saved re-evaluation summary to %s", summary_path)

    _wandb_log(
        {
            "reeval/overall_mean": summary["overall_mean"],
            "reeval/overall_std": summary["overall_std"],
            **{
                f"reeval/mean_{bid}": stats["mean"]
                for bid, stats in summary["buildings"].items()
                if stats["mean"] is not None
            },
        }
    )

    _save_best_config(best_params, output_dir / "configs", task)


# ── Main ─────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    task: str = cfg.task
    building_types: list[str] = list(cfg.get("building_types", BUILDING_TYPES))
    n_trials: int = int(cfg.get("n_trials", 100))
    n_startup_trials: int = int(cfg.get("n_startup_trials", 20))
    ntune_seeds: int = int(cfg.get("ntune_seeds", 3))
    n_tune_buildings: int = int(cfg.get("n_tune_buildings", 2))
    total_timesteps: int = int(cfg.training.total_timesteps)
    n_envs: int = int(cfg.training.n_envs)
    seed: int = int(cfg.get("seed", 0))
    is_reeval: bool = bool(cfg.get("reeval", False))
    is_analyze: bool = bool(cfg.get("analyze", False))
    wall_time_hours: float = float(cfg.get("wall_time_hours", 48))
    reeval_seeds: int = int(cfg.get("reeval_seeds", 30))
    reeval_timesteps: int = int(cfg.get("reeval_timesteps", 5_000_000))

    orion_db_dir = Path(str(cfg.get("orion_db_dir", "outputs/orion_dbs")))
    orion_db_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(str(cfg.get("results_dir", f"outputs/chs_results/all_{task}")))
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample buildings (deterministic, shared across all workers) ─
    # Each building is one CHS environment: training and evaluation happen
    # on the same building.  Buildings are drawn from the test split so
    # they are representative of the deployment distribution.
    rng = random.Random(seed)
    building_instances: list[tuple[str, str]] = []
    for bt in building_types:
        bids = _sample_one_per_cz(bt, "test", rng)
        if not bids:
            logger.warning("No test-split buildings found for %s; skipping.", bt)
            continue
        if len(bids) < n_tune_buildings:
            logger.warning(
                "%s: only %d buildings available in test split (requested %d)",
                bt,
                len(bids),
                n_tune_buildings,
            )
        for bid in bids[:n_tune_buildings]:
            building_instances.append((bt, bid))

    if not building_instances:
        logger.error("No buildings sampled across any building type.")
        return

    logger.info(
        "CHS PPO tuning for task=%s: %d buildings across %d types",
        task,
        len(building_instances),
        len(building_types),
    )
    for btype, bid in building_instances:
        logger.info("  %s / %s", btype, bid)

    # ── W&B init (if enabled) ────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(OmegaConf.select(wandb_cfg, "enabled", default=False))
    tb_log: str | None = None
    if use_wandb:
        try:
            import wandb

            mode_tag = "reeval" if is_reeval else ("analyze" if is_analyze else "sweep")
            if is_reeval:
                tb_log = str(output_dir / "tensorboard")
            wandb.init(
                project=OmegaConf.select(wandb_cfg, "project", default="b2b-baselines"),
                entity=OmegaConf.select(wandb_cfg, "entity", default=None),
                tags=list(OmegaConf.select(wandb_cfg, "tags", default=[])),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"chs_all_{task}_{mode_tag}",
                group=f"chs_all_{task}",
                sync_tensorboard=tb_log is not None,
            )
        except ImportError:
            logger.warning("wandb not installed; skipping init")
            use_wandb = False
            tb_log = None

    # ── Create / connect to shared Orion experiment ──────────────
    # All 75 SLURM workers share this single experiment regardless of
    # building type; Orion's pickledDB serialises concurrent access.
    exp_name = f"chs_ppo_all_{task}"
    db_path = orion_db_dir / f"{exp_name}.pkl"
    algorithm_cfg: str = str(cfg.get("algorithm", "tpe"))

    experiment = create_experiment(
        name=exp_name,
        space=ORION_SPACE,
        algorithm={
            algorithm_cfg: {
                "n_initial_points": n_startup_trials,
                "seed": seed,
            }
        },
        max_trials=n_trials,
        storage={
            "type": "legacy",
            "database": {
                "type": "pickleddb",
                "host": str(db_path),
            },
        },
    )

    if is_analyze:
        _run_analyze(experiment, results_dir, output_dir, task)
        return

    if is_reeval:
        _run_reeval(
            experiment,
            task,
            building_instances,
            total_timesteps=reeval_timesteps,
            n_envs=n_envs,
            reeval_seeds=reeval_seeds,
            results_dir=results_dir,
            output_dir=output_dir,
            tensorboard_log=tb_log,
        )
        return

    _run_sweep(
        experiment,
        task,
        building_instances,
        total_timesteps,
        n_envs,
        ntune_seeds,
        results_dir,
        wall_time_seconds=wall_time_hours * 3600,
    )


if __name__ == "__main__":
    main()
