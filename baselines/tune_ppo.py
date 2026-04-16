#!/usr/bin/env python3
"""Tune PPO hyperparameters with the CHS procedure (Patterson et al., RLC 2024).

For a given (building_type, task) pair, find the single best PPO
hyperparameter configuration by evaluating across buildings sampled
from different climate zones.  Uses Orion's masterless Service API
for distributed search via SLURM job arrays and CDF normalization
for post-hoc HP selection.

**Sweep mode** (default) -- run as a SLURM job array worker::

    python -m baselines.tune_ppo experiment=tune_ppo \\
        building_type=OfficeSmall task=task1

**CHS analysis** (after all sweep jobs complete)::

    python -m baselines.tune_ppo experiment=tune_ppo \\
        building_type=OfficeSmall task=task1 analyze=true

**Re-evaluation** (train+eval champion config with many seeds)::

    python -m baselines.tune_ppo experiment=tune_ppo \\
        building_type=OfficeSmall task=task1 reeval=true
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from orion.client import create_experiment
from stable_baselines3.common.monitor import Monitor

import building2building as b2b
from baselines.chs import CHSScoreStore, load_trial_rewards_from_dir
from baselines.utils.evaluation import run_episode
from baselines.utils.training import build_ppo, make_vec_env

logger = logging.getLogger(__name__)

PLACE_TO_CLIMATE_ZONE: dict[str, int] = {
    "Miami": 1,
    "Houston": 2,
    "Tampa": 2,
    "Tucson": 2,
    "Atlanta": 3,
    "ElPaso": 3,
    "SanDiego": 3,
    "SanFrancisco": 3,
    "Albuquerque": 4,
    "Baltimore": 4,
    "NewYork": 4,
    "PortAngeles": 4,
    "Seattle": 4,
    "Buffalo": 5,
    "Chicago": 5,
    "Denver": 5,
    "Vancouver": 5,
    "GreatFalls": 6,
    "Rochester": 6,
    "Duluth": 7,
    "InternationalFalls": 7,
    "Fairbanks": 8,
}

ORION_SPACE: dict[str, str] = {
    "/learning_rate": "loguniform(1e-5, 3e-4)",
    "/n_steps": "choices([256, 512, 1024, 2048])",
    "/batch_size": "choices([256, 512, 1024, 2048])",
    "/n_epochs": "uniform(3, 15, discrete=True)",
    "/ent_coef": "loguniform(1e-4, 0.05)",
    "/clip_range": "uniform(0.1, 0.4)",
    "/gae_lambda": "uniform(0.8, 1.0)",
    "/max_grad_norm": "uniform(0.3, 1.0)",
    "/vf_coef": "uniform(0.25, 1.0)",
    "/log_std_init": "uniform(-2.0, 0.0)",
    "/net_arch": "choices(['128_128', '256_256', '512_512'])",
}

_ARCH_MAP: dict[str, list[int]] = {
    "128_128": [128, 128],
    "256_256": [256, 256],
    "512_512": [512, 512],
}


# ── Climate zone helpers ─────────────────────────────────────────────


def _get_climate_zone(building_type: str, building_id: str) -> int | None:
    """Best-effort climate zone lookup from the building's weather file."""
    try:
        from building2building.data.registry import get_registry

        info = get_registry().get_building_by_id(building_type, building_id)
        weather = info.weather_file
        if not weather:
            return None
        place = Path(weather).stem.split("_")[0]
        return PLACE_TO_CLIMATE_ZONE.get(place)
    except Exception:
        return None


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


def _params_to_ppo_hparams(
    params: dict[str, Any], n_envs: int = 1
) -> dict[str, Any]:
    """Convert flat Orion trial params to PPO constructor kwargs.

    ``gamma`` is fixed at 0.99 (not tuned).  ``batch_size`` is clamped
    to ``n_steps * n_envs`` so SB3 never receives an impossible value.
    """
    arch = _ARCH_MAP[params["/net_arch"]]
    n_steps = int(params["/n_steps"])
    batch_size = min(int(params["/batch_size"]), n_steps * n_envs)
    return {
        "learning_rate": params["/learning_rate"],
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": int(params["/n_epochs"]),
        "ent_coef": params["/ent_coef"],
        "clip_range": params["/clip_range"],
        "gae_lambda": params["/gae_lambda"],
        "max_grad_norm": params["/max_grad_norm"],
        "vf_coef": params["/vf_coef"],
        "gamma": 0.99,
        "policy_kwargs": {
            "net_arch": {"pi": arch, "vf": arch},
            "activation_fn": "Tanh",
            "ortho_init": True,
            "log_std_init": params["/log_std_init"],
        },
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

    def make_train_env() -> Monitor:
        env = b2b.new_make_env(
            building_type, building_id=train_building_id, task=task
        )
        return Monitor(env)

    env_fns = [make_train_env for _ in range(n_envs)]
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

    eval_env = b2b.new_make_env(
        building_type, building_id=eval_building_id, task=task
    )
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
) -> None:
    """Write per-building rewards for one trial to a JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": trial_id,
        "trial_idx": trial_idx,
        "rewards": rewards,
    }
    out_path = results_dir / f"{trial_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))


# ── HP config I/O ────────────────────────────────────────────────────


def _trial_params_to_ppo_config(params: dict[str, Any]) -> dict[str, Any]:
    """Convert flat Orion trial params to a nested PPO config dict for YAML."""
    arch = _ARCH_MAP[params["/net_arch"]]
    return {
        "algorithm": "ppo",
        "policy_type": "MlpPolicy",
        "device": "auto",
        "learning_rate": float(params["/learning_rate"]),
        "n_steps": int(params["/n_steps"]),
        "batch_size": int(params["/batch_size"]),
        "n_epochs": int(params["/n_epochs"]),
        "ent_coef": float(params["/ent_coef"]),
        "clip_range": float(params["/clip_range"]),
        "gae_lambda": float(params["/gae_lambda"]),
        "max_grad_norm": float(params["/max_grad_norm"]),
        "vf_coef": float(params["/vf_coef"]),
        "gamma": 0.99,
        "target_kl": 0.02,
        "policy_kwargs": {
            "net_arch": {"pi": arch, "vf": arch},
            "activation_fn": "Tanh",
            "ortho_init": True,
            "log_std_init": float(params["/log_std_init"]),
        },
    }


def _save_best_config(
    params: dict[str, Any],
    output_dir: Path,
    building_type: str,
    task: str,
) -> Path:
    """Save the champion HP config as a YAML file."""
    cfg = _trial_params_to_ppo_config(params)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ppo_chs_{building_type.lower()}_{task}.yaml"
    out_path = output_dir / fname
    out_path.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    )
    logger.info("Saved CHS-tuned PPO config to %s", out_path)
    return out_path


# ── Orion worker loop ────────────────────────────────────────────────


def _run_sweep(
    experiment: Any,
    building_type: str,
    task: str,
    train_building_ids: list[str],
    eval_building_ids: list[str],
    total_timesteps: int,
    n_envs: int,
    ntune_seeds: int,
    results_dir: Path,
) -> None:
    """Orion worker loop: suggest trials, train, observe, repeat.

    TensorBoard logging is intentionally disabled during the sweep to
    avoid filling up disk quota across many short training runs.
    """
    trial_counter = 0
    while not experiment.is_done:
        trial = experiment.suggest()
        if trial is None:
            logger.info("No more trials to suggest; worker exiting.")
            break

        params = trial.params
        hparams = _params_to_ppo_hparams(params, n_envs=n_envs)
        trial_idx = trial_counter
        trial_counter += 1

        logger.info(
            "Worker starting trial %s (idx %d)", trial.id, trial_idx
        )

        all_rewards: list[float] = []
        rewards_per_building: dict[str, list[float]] = {}
        failed = False

        for train_bid, eval_bid in zip(
            train_building_ids, eval_building_ids
        ):
            rewards_per_building[eval_bid] = []
            for seed in range(ntune_seeds):
                try:
                    reward = _train_and_eval_single(
                        building_type,
                        train_bid,
                        eval_bid,
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
                        "Trial %s seed %d failed on %s -> %s: %s",
                        trial.id,
                        seed,
                        train_bid,
                        eval_bid,
                        exc,
                    )
                    failed = True
                    break

                rewards_per_building[eval_bid].append(reward)
                all_rewards.append(reward)
                logger.info(
                    "Trial %s | %s -> %s | seed %d | reward %.1f",
                    trial.id,
                    train_bid,
                    eval_bid,
                    seed,
                    reward,
                )
            if failed:
                break

        if failed or len(all_rewards) == 0:
            objective_value = 1e10
        else:
            objective_value = -float(np.mean(all_rewards))

        experiment.observe(
            trial,
            [{"name": "objective", "type": "objective", "value": objective_value}],
        )

        _save_trial_rewards(
            trial.id, trial_idx, rewards_per_building, results_dir
        )
        logger.info(
            "Trial %s observed (objective=%.1f)", trial.id, objective_value
        )

        _wandb_log({
            "trial/index": trial_idx,
            "trial/mean_reward": -objective_value if objective_value < 1e9 else float("nan"),
            "trial/objective": objective_value,
            "trial/failed": failed,
            **{
                f"trial/reward_{bid}": float(np.mean(rews)) if rews else float("nan")
                for bid, rews in rewards_per_building.items()
            },
        })


# ── CHS analysis ────────────────────────────────────────────────────


def _run_analyze(
    experiment: Any,
    results_dir: Path,
    output_dir: Path,
    building_type: str,
    task: str,
) -> None:
    """Post-hoc CHS analysis: CDF-normalize and select the best trial."""
    store = load_trial_rewards_from_dir(results_dir)
    if not store.env_ids():
        logger.error("No trial reward files found in %s", results_dir)
        return

    trial_ids = store.trial_ids()
    n_trials = len(trial_ids)
    logger.info(
        "Loaded rewards for %d trials across %d buildings",
        n_trials,
        len(store.env_ids()),
    )

    ranking = store.trial_summary(trial_ids)
    logger.info("CHS ranking (top 5):")
    for rank, (tid, cdf_score) in enumerate(ranking[:5], 1):
        logger.info("  #%d  trial %s  CDF score %.4f", rank, tid, cdf_score)

    chs_best_id = ranking[0][0]

    # Retrieve the best trial's params from Orion.
    best_params: dict[str, Any] | None = None
    for trial in experiment.fetch_trials():
        if trial.id == chs_best_id:
            best_params = trial.params
            break

    if best_params is None:
        logger.error(
            "Could not find trial %s in Orion experiment.", chs_best_id
        )
        return

    _save_best_config(
        best_params, output_dir / "configs", building_type, task
    )
    logger.info("CHS analysis complete. Best trial: %s", chs_best_id)


# ── Re-evaluation ────────────────────────────────────────────────────


def _run_reeval(
    experiment: Any,
    building_type: str,
    task: str,
    eval_building_ids: list[str],
    total_timesteps: int,
    n_envs: int,
    reeval_seeds: int,
    results_dir: Path,
    output_dir: Path,
    tensorboard_log: str | None = None,
) -> None:
    """Re-evaluate the CHS-best trial with many seeds on eval buildings."""
    store = load_trial_rewards_from_dir(results_dir)
    trial_ids = store.trial_ids()
    if not trial_ids:
        logger.error("No trial reward files found; run sweep + analyze first.")
        return

    ranking = store.trial_summary(trial_ids)
    chs_best_id = ranking[0][0]

    best_params: dict[str, Any] | None = None
    for trial in experiment.fetch_trials():
        if trial.id == chs_best_id:
            best_params = trial.params
            break

    if best_params is None:
        logger.error(
            "Could not find trial %s in Orion experiment.", chs_best_id
        )
        return

    hparams = _params_to_ppo_hparams(best_params, n_envs=n_envs)
    logger.info(
        "Re-evaluating CHS-best trial %s on %d buildings x %d seeds",
        chs_best_id,
        len(eval_building_ids),
        reeval_seeds,
    )

    results: dict[str, list[float]] = {}
    for eval_bid in eval_building_ids:
        results[eval_bid] = []
        for seed in range(reeval_seeds):
            try:
                reward = _train_and_eval_single(
                    building_type,
                    eval_bid,
                    eval_bid,
                    task,
                    hparams,
                    total_timesteps,
                    n_envs,
                    seed=seed,
                    tensorboard_log=tensorboard_log,
                )
                results[eval_bid].append(reward)
                logger.info(
                    "Reeval %s seed %d: reward %.1f",
                    eval_bid,
                    seed,
                    reward,
                )
                _wandb_log({
                    "reeval/building": eval_bid,
                    "reeval/seed": seed,
                    "reeval/reward": reward,
                })
            except Exception:
                logger.exception(
                    "Reeval failed on %s seed %d", eval_bid, seed
                )

    reeval_dir = output_dir / "reeval"
    reeval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reeval_dir / f"reeval_{building_type}_{task}.yaml"

    summary: dict[str, Any] = {
        "building_type": building_type,
        "task": task,
        "best_trial": chs_best_id,
        "reeval_seeds": reeval_seeds,
        "total_timesteps": total_timesteps,
        "buildings": {},
    }
    all_rewards: list[float] = []
    for bid, rewards in results.items():
        arr = np.array(rewards)
        all_rewards.extend(rewards)
        summary["buildings"][bid] = {
            "mean": float(arr.mean()) if len(arr) > 0 else None,
            "std": float(arr.std()) if len(arr) > 0 else None,
            "n": len(rewards),
            "rewards": [float(r) for r in rewards],
        }

    all_arr = np.array(all_rewards)
    summary["overall_mean"] = (
        float(all_arr.mean()) if len(all_arr) > 0 else None
    )
    summary["overall_std"] = (
        float(all_arr.std()) if len(all_arr) > 0 else None
    )

    summary_path.write_text(
        yaml.dump(summary, default_flow_style=False, sort_keys=False)
    )
    logger.info("Saved re-evaluation summary to %s", summary_path)

    _wandb_log({
        "reeval/overall_mean": summary["overall_mean"],
        "reeval/overall_std": summary["overall_std"],
        **{
            f"reeval/mean_{bid}": stats["mean"]
            for bid, stats in summary["buildings"].items()
            if stats["mean"] is not None
        },
    })

    _save_best_config(
        best_params, output_dir / "configs", building_type, task
    )


# ── Main ─────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    building_type: str = cfg.building_type
    task: str = cfg.task
    n_trials: int = int(cfg.get("n_trials", 100))
    n_startup_trials: int = int(cfg.get("n_startup_trials", 20))
    ntune_seeds: int = int(cfg.get("ntune_seeds", 3))
    n_tune_buildings: int | None = (
        int(cfg.n_tune_buildings) if cfg.get("n_tune_buildings") is not None else None
    )
    total_timesteps: int = int(cfg.training.total_timesteps)
    n_envs: int = int(cfg.training.n_envs)
    seed: int = int(cfg.get("seed", 0))
    is_reeval: bool = bool(cfg.get("reeval", False))
    is_analyze: bool = bool(cfg.get("analyze", False))
    reeval_seeds: int = int(cfg.get("reeval_seeds", 30))
    reeval_timesteps: int = int(cfg.get("reeval_timesteps", 5_000_000))

    orion_db_dir = Path(str(cfg.get("orion_db_dir", "outputs/orion_dbs")))
    orion_db_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(str(cfg.get("results_dir", f"outputs/chs_results/{building_type}_{task}")))
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample buildings (deterministic, same across all workers) ─
    rng = random.Random(seed)
    train_building_ids = _sample_one_per_cz(building_type, "train", rng)
    eval_building_ids = _sample_one_per_cz(building_type, "test", rng)

    if not train_building_ids or not eval_building_ids:
        logger.error(
            "Could not sample buildings for %s. "
            "Check that data is available for both splits.",
            building_type,
        )
        return

    n_buildings = min(len(train_building_ids), len(eval_building_ids))
    train_building_ids = train_building_ids[:n_buildings]
    eval_building_ids = eval_building_ids[:n_buildings]

    logger.info(
        "CHS PPO tuning for %s / %s (%d buildings sampled)",
        building_type,
        task,
        n_buildings,
    )
    logger.info("Train buildings: %s", train_building_ids)
    logger.info("Eval buildings:  %s", eval_building_ids)

    # ── W&B init (if enabled) ────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(OmegaConf.select(wandb_cfg, "enabled", default=False))
    tb_log: str | None = None
    if use_wandb:
        try:
            import wandb

            mode_tag = "reeval" if is_reeval else ("analyze" if is_analyze else "sweep")
            # Only enable TensorBoard sync for reeval (sweep uses
            # wandb.log directly to avoid filling disk quota).
            if is_reeval:
                tb_log = str(output_dir / "tensorboard")
            wandb.init(
                project=OmegaConf.select(
                    wandb_cfg, "project", default="b2b-baselines"
                ),
                entity=OmegaConf.select(wandb_cfg, "entity", default=None),
                tags=list(OmegaConf.select(wandb_cfg, "tags", default=[])),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"chs_{building_type}_{task}_{mode_tag}",
                group=f"chs_{building_type}_{task}",
                sync_tensorboard=tb_log is not None,
            )
        except ImportError:
            logger.warning("wandb not installed; skipping init")
            use_wandb = False
            tb_log = None

    # ── Create / connect to Orion experiment ─────────────────────
    exp_name = f"chs_ppo_{building_type.lower()}_{task}"
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

    # ── Dispatch to the requested mode ───────────────────────────
    if is_analyze:
        _run_analyze(
            experiment, results_dir, output_dir, building_type, task
        )
        return

    if is_reeval:
        _run_reeval(
            experiment,
            building_type,
            task,
            eval_building_ids,
            total_timesteps=reeval_timesteps,
            n_envs=n_envs,
            reeval_seeds=reeval_seeds,
            results_dir=results_dir,
            output_dir=output_dir,
            tensorboard_log=tb_log,
        )
        return

    # ── Default: sweep worker loop ───────────────────────────────
    sweep_train_ids = train_building_ids
    sweep_eval_ids = eval_building_ids
    if n_tune_buildings is not None and n_tune_buildings < len(sweep_train_ids):
        sweep_train_ids = train_building_ids[:n_tune_buildings]
        sweep_eval_ids = eval_building_ids[:n_tune_buildings]
        logger.info(
            "Sweep limited to %d buildings (of %d available)",
            n_tune_buildings,
            len(train_building_ids),
        )

    _run_sweep(
        experiment,
        building_type,
        task,
        sweep_train_ids,
        sweep_eval_ids,
        total_timesteps,
        n_envs,
        ntune_seeds,
        results_dir,
    )


if __name__ == "__main__":
    main()
