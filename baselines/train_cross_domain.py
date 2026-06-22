#!/usr/bin/env python3
"""Cross-domain RL training for Building2Building.

Trains a single Amorpheus policy across buildings with different
morphologies using PPO with Beta-distributed actions, GAE, KL early
stopping, and multi-worker rollout collection.

The model architecture lives in :mod:`baselines.models.amorpheus`.

Usage:
    python -m baselines.train_cross_domain
    python -m baselines.train_cross_domain training.n_buildings_per_type=3
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import building2building as b2b
from building2building.morphology import Morphology
from baselines.models.amorpheus import (
    BUILDING_TYPE_INDEX,
    N_BUILDING_TYPES,
    ExtensionConfig,
    Model,
    Policy,
    encode_observation,
    join_action,
    make_building_type_onehot,
    obs_to_tensors,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Rollout storage
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Fixed-length rollout segment from one building."""

    policy: Policy
    # Per-node local obs, stored as a list of (T, obs_dim_i) arrays
    local_obs: list[np.ndarray]
    # Flat actions in physical units over all action slots (node order)
    actions: np.ndarray  # (T, total_action_dim)
    rewards: np.ndarray  # (T,)
    values: np.ndarray  # (T,)
    log_probs: np.ndarray  # (T,)
    dones: np.ndarray  # (T,)
    last_value: float
    building_type_index: int


# ---------------------------------------------------------------------------
# 2. Building pool
# ---------------------------------------------------------------------------


@dataclass
class BuildingSpec:
    building_type: str
    split: str
    index: int
    building_id: str


class BuildingPool:
    """Manages a pool of buildings for diverse training rollouts."""

    def __init__(
        self,
        building_types: list[str],
        n_per_type: int,
        split: str = "train",
        task: str = "task1",
    ):
        self.specs: list[BuildingSpec] = []
        self.task = task

        for btype in building_types:
            ids = b2b.list_buildings(btype, split)[:n_per_type]
            for i, bid in enumerate(ids):
                self.specs.append(
                    BuildingSpec(
                        building_type=btype,
                        split=split,
                        index=i,
                        building_id=bid,
                    )
                )

        logger.info(
            "BuildingPool: %d buildings across %s",
            len(self.specs),
            building_types,
        )

    def make_env(self, spec: BuildingSpec, eplus_output_dir: Path) -> gym.Env:
        return b2b.make_env(
            spec.building_type,
            split=spec.split,
            building_id=spec.building_id,
            task=self.task,
            eplus_output_dir=eplus_output_dir,
        )

    def sample_batch(self, rng: np.random.Generator) -> list[BuildingSpec]:
        """Sample exactly one random building per type."""
        by_type: dict[str, list[BuildingSpec]] = {}
        for spec in self.specs:
            by_type.setdefault(spec.building_type, []).append(spec)
        return [specs[int(rng.integers(len(specs)))] for specs in by_type.values()]


# ---------------------------------------------------------------------------
# 3. Rollout collection
# ---------------------------------------------------------------------------


@dataclass
class EnvState:
    env: gym.Env
    morphology: Morphology
    policy: Policy
    obs: np.ndarray
    building_type: str


@torch.no_grad()
def collect_n_steps(
    env_state: EnvState,
    device: torch.device,
    n_steps: int,
) -> tuple[Trajectory, list[float]]:
    """Collect exactly n_steps transitions, auto-resetting on done."""
    morphology = env_state.morphology
    nodes = morphology.nodes
    all_local_obs: list[list[np.ndarray]] = [[] for _ in nodes]
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []
    values_list: list[float] = []
    log_probs_list: list[float] = []
    dones_list: list[float] = []
    completed_episode_rewards: list[float] = []
    current_episode_reward = 0.0

    obs = env_state.obs
    env = env_state.env
    policy = env_state.policy
    bt_onehot = make_building_type_onehot(env_state.building_type, device)

    for _ in range(n_steps):
        local_obs = encode_observation(morphology, obs)
        local_tensors = obs_to_tensors(local_obs, device)

        dist, value = policy.forward(local_tensors, bt_onehot)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t).sum(dim=-1)

        action_np = action_t.squeeze(0).cpu().numpy()
        env_action = join_action(morphology, action_np)

        obs2, reward, terminated, truncated, _info = env.step(env_action)
        done = bool(terminated or truncated)

        for i, node_obs in enumerate(local_obs):
            all_local_obs[i].append(node_obs)
        actions_list.append(action_np)
        rewards_list.append(reward)
        values_list.append(value.item())
        log_probs_list.append(log_prob.item())
        dones_list.append(float(done))
        current_episode_reward += reward

        if done:
            completed_episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs2, _info = env.reset()

        obs = obs2

    # Bootstrap value
    if dones_list[-1] > 0.5:
        last_value = 0.0
    else:
        local_obs = encode_observation(morphology, obs)
        local_tensors = obs_to_tensors(local_obs, device)
        _, bootstrap_v = policy.forward(local_tensors, bt_onehot)
        last_value = bootstrap_v.item()

    env_state.obs = obs

    bt_idx = BUILDING_TYPE_INDEX.get(env_state.building_type, 0)

    return (
        Trajectory(
            policy=policy,
            local_obs=[np.stack(node_list) for node_list in all_local_obs],
            actions=np.stack(actions_list),
            rewards=np.array(rewards_list, dtype=np.float32),
            values=np.array(values_list, dtype=np.float32),
            log_probs=np.array(log_probs_list, dtype=np.float32),
            dones=np.array(dones_list, dtype=np.float32),
            last_value=last_value,
            building_type_index=bt_idx,
        ),
        completed_episode_rewards,
    )


# ---------------------------------------------------------------------------
# 4. GAE and PPO update
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1].item()
        next_non_terminal = 1.0 - dones[t].item()
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae
        )
    return advantages, advantages + values


def ppo_update(
    optimizer: torch.optim.Optimizer,
    trajectories: list[Trajectory],
    device: torch.device,
    *,
    n_epochs: int = 5,
    minibatch_size: int = 336,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    gamma: float = 0.98,
    gae_lambda: float = 0.95,
    target_kl: float | None = 0.02,
) -> dict[str, float]:
    """PPO update.  Each trajectory has its own Policy (different morphology)."""
    PreparedData = tuple[
        Policy,
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    all_data: list[PreparedData] = []

    for traj in trajectories:
        local_obs_t = [torch.tensor(x, device=device) for x in traj.local_obs]
        actions = torch.tensor(traj.actions, device=device)
        old_logprobs = torch.tensor(traj.log_probs, device=device)
        rewards = torch.tensor(traj.rewards, device=device)
        values = torch.tensor(traj.values, device=device)
        dones = torch.tensor(traj.dones, device=device)

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            last_value=traj.last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        bt = torch.zeros(N_BUILDING_TYPES, device=device)
        bt[traj.building_type_index] = 1.0

        all_data.append(
            (traj.policy, local_obs_t, actions, old_logprobs, advantages, returns, bt)
        )

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }
    n_updates = 0
    kl_exceeded = False

    for _epoch in range(n_epochs):
        if kl_exceeded:
            break
        for (
            policy,
            local_obs_t,
            actions,
            old_logprobs,
            advantages,
            returns,
            bt_onehot,
        ) in all_data:
            if kl_exceeded:
                break
            T = len(advantages)
            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            indices = torch.randperm(T, device=device)

            for start in range(0, T, minibatch_size):
                end = min(start + minibatch_size, T)
                mb_idx = indices[start:end]
                mb_size = end - start

                mb_local = [obs_t[mb_idx] for obs_t in local_obs_t]
                mb_bt = bt_onehot.unsqueeze(0).expand(mb_size, -1)
                mb_actions = actions[mb_idx]
                mb_old_lp = old_logprobs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = returns[mb_idx]

                dist, new_value = policy.forward(mb_local, mb_bt)
                new_logprob = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                log_ratio = new_logprob - mb_old_lp
                ratio = log_ratio.exp()
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps),
                ).mean()

                v_loss = F.mse_loss(new_value, mb_ret)
                ent = entropy.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent

                optimizer.zero_grad()
                loss.backward()
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                optimizer.step()

                stats["policy_loss"] += pg_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["total_loss"] += loss.item()
                n_updates += 1

                if target_kl is not None:
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    if approx_kl > 1.5 * target_kl:
                        kl_exceeded = True
                        break

    if n_updates > 0:
        for k in stats:
            stats[k] /= n_updates
    return stats


# ---------------------------------------------------------------------------
# 5. Rollout workers (subprocess)
# ---------------------------------------------------------------------------


@dataclass
class RolloutJob:
    spec: BuildingSpec
    eplus_dir: Path
    model_kwargs: dict[str, Any]
    state_dict: dict
    n_steps: int
    task: str


def _run_rollout_job(job: RolloutJob) -> tuple:
    """Create env, collect n_steps, return arrays for reconstruction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.set_num_threads(1)

    t_start = time.time()

    env = b2b.make_env(
        job.spec.building_type,
        split=job.spec.split,
        building_id=job.spec.building_id,
        task=job.task,
        eplus_output_dir=job.eplus_dir,
    )
    morphology: Morphology = env.metadata["morphology"]
    model = Model(**job.model_kwargs)
    model.load_state_dict(job.state_dict)
    model.eval()
    policy = model.condition(morphology)

    obs, _info = env.reset()
    t_env_ready = time.time()

    env_state = EnvState(
        env=env,
        morphology=morphology,
        policy=policy,
        obs=obs,
        building_type=job.spec.building_type,
    )

    traj, completed = collect_n_steps(env_state, torch.device("cpu"), job.n_steps)
    t_collect_done = time.time()

    env.close()

    env_setup_s = t_env_ready - t_start
    collect_s = t_collect_done - t_env_ready

    return (
        job.spec.building_id,
        morphology,
        job.spec.building_type,
        traj.local_obs,
        traj.actions,
        traj.rewards,
        traj.values,
        traj.log_probs,
        traj.dones,
        traj.last_value,
        traj.building_type_index,
        completed,
        env_setup_s,
        collect_s,
    )


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalJob:
    spec: BuildingSpec
    eplus_dir: Path
    model_kwargs: dict[str, Any]
    state_dict: dict
    task: str


def _run_eval_job(job: EvalJob) -> tuple[str, str, float, int]:
    """Run one full episode.  Returns (building_type, building_id, reward, steps)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.set_num_threads(1)

    env = b2b.make_env(
        job.spec.building_type,
        split=job.spec.split,
        building_id=job.spec.building_id,
        task=job.task,
        eplus_output_dir=job.eplus_dir,
    )
    morphology: Morphology = env.metadata["morphology"]
    model = Model(**job.model_kwargs)
    model.load_state_dict(job.state_dict)
    model.eval()
    policy = model.condition(morphology)

    device = torch.device("cpu")
    bt_onehot = make_building_type_onehot(job.spec.building_type, device)
    obs, _info = env.reset()
    done = False
    total_reward = 0.0
    n_steps = 0

    with torch.no_grad():
        while not done:
            local_obs = encode_observation(morphology, obs)
            local_tensors = obs_to_tensors(local_obs, device)
            dist, _ = policy.forward(local_tensors, bt_onehot)
            assert dist is not None
            action_np = dist.mean.squeeze(0).cpu().numpy()
            env_action = join_action(morphology, action_np)
            obs, reward, terminated, truncated, _info = env.step(env_action)
            done = bool(terminated or truncated)
            total_reward += reward
            n_steps += 1

    env.close()
    return (job.spec.building_type, job.spec.building_id, total_reward, n_steps)


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3

    total_iterations: int = 200
    n_steps: int = 672
    max_workers: int = 8
    learning_rate: float = 5e-5
    gamma: float = 0.98
    gae_lambda: float = 0.95
    n_ppo_epochs: int = 5
    minibatch_size: int = 336
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    seed: int = 42

    building_types: list[str] = field(
        default_factory=lambda: ["RetailStandalone", "OfficeSmall"]
    )
    n_buildings_per_type: int = 10
    resample_interval: int = 10
    split: str = "train"
    task: str = "task1"

    output_dir: str = "outputs/transfer"
    log_interval: int = 1
    save_interval: int = 50
    eval_interval: int = 50

    resume_from: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eplus_dir = output_dir / "eplus_outputs"

    logger.info("Device: %s", device)
    logger.info("Building types: %s", cfg.building_types)

    pool = BuildingPool(
        building_types=cfg.building_types,
        n_per_type=cfg.n_buildings_per_type,
        split=cfg.split,
        task=cfg.task,
    )

    model_kwargs = dict(d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers)
    model = Model(**model_kwargs).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    start_iteration = 1
    if cfg.resume_from is not None:
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt["iteration"] + 1
        logger.info("Resumed from iteration %d", start_iteration - 1)

    wandb_run = None
    try:
        import wandb

        if cfg.wandb_project:
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=vars(cfg),
            )
    except ImportError:
        pass

    eval_pool: BuildingPool | None = None
    if cfg.eval_interval > 0:
        eval_pool = BuildingPool(
            building_types=cfg.building_types,
            n_per_type=1,
            split="test",
            task=cfg.task,
        )

    spawn_ctx = mp.get_context("spawn")
    logger.info(
        "%d buildings, %d steps/building/iter, max %d workers",
        len(pool.specs),
        cfg.n_steps,
        cfg.max_workers,
    )

    # Cache conditioned policies by (building_id, building_type)
    policies: dict[str, Policy] = {}
    current_batch: list[BuildingSpec] = []

    for iteration in range(start_iteration, cfg.total_iterations + 1):
        t0 = time.time()
        model.eval()

        if (
            not current_batch
            or (iteration - start_iteration) % cfg.resample_interval == 0
        ):
            current_batch = pool.sample_batch(rng)
            logger.info(
                "iter %d | resampled: %s",
                iteration,
                [(s.building_type, s.building_id) for s in current_batch],
            )

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        jobs = [
            RolloutJob(
                spec=spec,
                eplus_dir=eplus_dir / f"iter_{iteration}",
                model_kwargs=model_kwargs,
                state_dict=state_dict,
                n_steps=cfg.n_steps,
                task=cfg.task,
            )
            for spec in current_batch
        ]

        with spawn_ctx.Pool(processes=min(cfg.max_workers, len(jobs))) as wp:
            results = wp.map(_run_rollout_job, jobs)

        trajectories: list[Trajectory] = []
        ep_rewards: list[float] = []
        env_setup_times: list[float] = []
        collect_times: list[float] = []
        for (
            building_id,
            morphology,
            building_type,
            local_obs,
            actions,
            rewards,
            values,
            log_probs,
            dones,
            last_value,
            bt_idx,
            completed,
            env_setup_s,
            collect_s,
        ) in results:
            if building_id not in policies:
                policies[building_id] = model.condition(morphology)
            trajectories.append(
                Trajectory(
                    policy=policies[building_id],
                    local_obs=local_obs,
                    actions=actions,
                    rewards=rewards,
                    values=values,
                    log_probs=log_probs,
                    dones=dones,
                    last_value=last_value,
                    building_type_index=bt_idx,
                )
            )
            ep_rewards.extend(completed)
            env_setup_times.append(env_setup_s)
            collect_times.append(collect_s)

        model.train()
        stats = ppo_update(
            optimizer,
            trajectories,
            device,
            n_epochs=cfg.n_ppo_epochs,
            minibatch_size=cfg.minibatch_size,
            clip_eps=cfg.clip_eps,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
            max_grad_norm=cfg.max_grad_norm,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            target_kl=cfg.target_kl,
        )

        t_ppo_done = time.time()
        elapsed = t_ppo_done - t0
        ppo_s = (
            t_ppo_done - t0 - max(s + c for s, c in zip(env_setup_times, collect_times))
            if env_setup_times
            else 0.0
        )
        if iteration % cfg.log_interval == 0:
            mean_r = np.mean([t.rewards.mean() for t in trajectories])
            ep_str = f"{np.mean(ep_rewards):.1f}" if ep_rewards else "n/a"
            mean_setup = np.mean(env_setup_times)
            mean_collect = np.mean(collect_times)
            logger.info(
                "iter %4d | r/step %.3f | ep_reward %s (%d eps) | "
                "pg %.4f | v %.4f | ent %.4f | "
                "env_setup %.1fs | collect %.1fs | ppo %.1fs | total %.1fs",
                iteration,
                mean_r,
                ep_str,
                len(ep_rewards),
                stats["policy_loss"],
                stats["value_loss"],
                stats["entropy"],
                mean_setup,
                mean_collect,
                ppo_s,
                elapsed,
            )
            if wandb_run is not None:
                log_d: dict[str, float] = {
                    "reward_per_step": float(mean_r),
                    **{k: v for k, v in stats.items()},
                    "time_s": elapsed,
                }
                if ep_rewards:
                    log_d["episode_reward"] = float(np.mean(ep_rewards))
                wandb_run.log(log_d, step=iteration)

        if iteration % cfg.save_interval == 0:
            ckpt_path = output_dir / f"model_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info("Saved: %s", ckpt_path)

        if (
            eval_pool is not None
            and cfg.eval_interval > 0
            and iteration % cfg.eval_interval == 0
        ):
            logger.info(
                "iter %d | eval on %d test buildings",
                iteration,
                len(eval_pool.specs),
            )
            eval_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            eval_jobs = [
                EvalJob(
                    spec=s,
                    eplus_dir=eplus_dir / f"eval_{iteration}",
                    model_kwargs=model_kwargs,
                    state_dict=eval_sd,
                    task=cfg.task,
                )
                for s in eval_pool.specs
            ]
            with spawn_ctx.Pool(processes=len(eval_jobs)) as ep:
                eval_results = ep.map(_run_eval_job, eval_jobs)

            for bt, bid, rew, steps in eval_results:
                logger.info("  %s %s: reward=%.1f steps=%d", bt, bid, rew, steps)
            mean_eval = float(np.mean([r for _, _, r, _ in eval_results]))
            logger.info("iter %d | eval mean: %.1f", iteration, mean_eval)
            if wandb_run is not None:
                wandb_run.log({"eval/mean_reward": mean_eval}, step=iteration)

    torch.save(
        {
            "iteration": cfg.total_iterations,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
        output_dir / "model_final.pt",
    )
    logger.info("Done. Final model: %s", output_dir / "model_final.pt")

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# 8. Hydra entry point
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        import hydra
        from omegaconf import DictConfig, OmegaConf

        @hydra.main(version_base=None, config_path="configs", config_name="config")
        def hydra_main(cfg: DictConfig) -> None:
            logging.basicConfig(level=logging.INFO)
            raw = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(raw, dict)

            model_cfg = raw.get("model", {})
            training_cfg = raw.get("training", {})

            train(
                TrainConfig(
                    d_model=int(model_cfg.get("d_model", 64)),
                    n_heads=int(model_cfg.get("n_heads", 4)),
                    n_layers=int(model_cfg.get("n_layers", 3)),
                    total_iterations=int(training_cfg.get("total_iterations", 200)),
                    n_steps=int(training_cfg.get("n_steps", 672)),
                    max_workers=int(training_cfg.get("max_workers", 8)),
                    learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
                    gamma=float(training_cfg.get("gamma", 0.98)),
                    gae_lambda=float(training_cfg.get("gae_lambda", 0.95)),
                    n_ppo_epochs=int(training_cfg.get("n_ppo_epochs", 5)),
                    minibatch_size=int(training_cfg.get("minibatch_size", 336)),
                    clip_eps=float(training_cfg.get("clip_eps", 0.2)),
                    vf_coef=float(training_cfg.get("vf_coef", 0.5)),
                    ent_coef=float(training_cfg.get("ent_coef", 0.01)),
                    max_grad_norm=float(training_cfg.get("max_grad_norm", 0.5)),
                    target_kl=training_cfg.get("target_kl", 0.02),
                    seed=int(training_cfg.get("seed", 42)),
                    building_types=list(
                        training_cfg.get(
                            "building_types",
                            ["RetailStandalone", "OfficeSmall"],
                        )
                    ),
                    n_buildings_per_type=int(
                        training_cfg.get("n_buildings_per_type", 10)
                    ),
                    resample_interval=int(training_cfg.get("resample_interval", 10)),
                    split=str(training_cfg.get("split", "train")),
                    task=str(raw.get("task", "task1")),
                    output_dir=str(raw.get("output_dir", "outputs/transfer")),
                    log_interval=int(training_cfg.get("log_interval", 1)),
                    save_interval=int(training_cfg.get("save_interval", 50)),
                    eval_interval=int(training_cfg.get("eval_interval", 50)),
                    resume_from=raw.get("resume_from"),
                    wandb_project=raw.get("wandb_project"),
                    wandb_entity=raw.get("wandb_entity"),
                )
            )

        hydra_main()

    except ImportError:
        logging.basicConfig(level=logging.INFO)
        train(TrainConfig())


if __name__ == "__main__":
    main()
