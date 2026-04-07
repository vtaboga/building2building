#!/usr/bin/env python3
"""Train Amorpheus on a cross-domain benchmark preset (Section 6.2).

Uses the ``CrossDomainGeneralization`` benchmark for single train-type ->
test-type transfer, with a standalone PPO training loop.

Usage with Hydra::

    python -m baselines.train_cross_domain experiment=train_cross_domain
    python -m baselines.train_cross_domain experiment=train_cross_domain \
        total_timesteps=5_000_000 model.embed_dim=128
"""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

import building2building as b2b
from baselines.models.amorpheus import AmorpheusPolicy
from baselines.utils.evaluation import run_episode

logger = logging.getLogger(__name__)


def _make_env(building_type: str, index: int, task: str) -> gym.Env:
    return b2b.new_make_env(
        building_type, split="train", index=index, task=task
    )


def _get_morphology(building_type: str, task: str) -> b2b.Morphology:
    env = _make_env(building_type, 0, task)
    morph: b2b.Morphology = env.metadata["morphology"]
    env.close()
    return morph


def collect_rollout(
    env: gym.Env,
    policy: AmorpheusPolicy,
    n_steps: int,
) -> dict[str, torch.Tensor]:
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    rew_list: list[float] = []
    done_list: list[bool] = []
    val_list: list[float] = []

    obs, _ = env.reset()

    for _ in range(n_steps):
        obs_t = torch.from_numpy(
            np.asarray(obs, dtype=np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            actions, values = policy(obs_t)

        action_np = actions.squeeze(0).numpy()
        obs_list.append(obs)
        act_list.append(action_np)
        val_list.append(values.item())

        obs, reward, terminated, truncated, _info = env.step(action_np)
        rew_list.append(float(reward))
        done_list.append(terminated or truncated)

        if terminated or truncated:
            obs, _ = env.reset()

    return {
        "obs": torch.from_numpy(np.array(obs_list, dtype=np.float32)),
        "actions": torch.from_numpy(np.array(act_list, dtype=np.float32)),
        "rewards": torch.tensor(rew_list, dtype=torch.float32),
        "dones": torch.tensor(done_list, dtype=torch.float32),
        "values": torch.tensor(val_list, dtype=torch.float32),
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(rewards)
    advantages = torch.zeros(n)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: AmorpheusPolicy,
    optimizer: optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    *,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    n_epochs: int = 5,
    batch_size: int = 64,
) -> dict[str, float]:
    obs = rollout["obs"]
    actions_old = rollout["actions"]
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values_old = rollout["values"]

    advantages, returns = compute_gae(rewards, values_old, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    with torch.no_grad():
        actions_pred, _ = policy(obs)
        old_log_probs = -0.5 * ((actions_old - actions_pred) ** 2).sum(dim=-1)

    total_loss_sum = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        indices = torch.randperm(len(obs))
        for start in range(0, len(obs), batch_size):
            end = min(start + batch_size, len(obs))
            idx = indices[start:end]

            batch_obs = obs[idx]
            batch_act_old = actions_old[idx]
            batch_adv = advantages[idx]
            batch_ret = returns[idx]
            batch_old_lp = old_log_probs[idx]

            actions_new, values_new = policy(batch_obs)
            new_log_probs = -0.5 * (
                (batch_act_old - actions_new) ** 2
            ).sum(dim=-1)

            ratio = (new_log_probs - batch_old_lp).exp()
            surr1 = ratio * batch_adv
            surr2 = (
                torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * batch_adv
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(
                values_new.squeeze(-1), batch_ret
            )
            loss = policy_loss + vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss_sum += loss.item()
            n_updates += 1

    return {"loss": total_loss_sum / max(n_updates, 1)}


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    task: str = cfg.reward.task_name
    total_timesteps: int = int(cfg.total_timesteps)
    rollout_steps: int = int(cfg.rollout_steps)
    seed: int = int(cfg.seed)

    model_cfg = cfg.model
    embed_dim: int = int(model_cfg.embed_dim)
    n_heads: int = int(model_cfg.n_heads)
    n_layers: int = int(model_cfg.n_layers)
    lr: float = float(model_cfg.lr)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_types: list[str] = list(cfg.train_building_types)
    test_per_type: int = int(cfg.get("test_buildings_per_type", 5))

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(
        "Training Amorpheus on %d building types: %s",
        len(train_types),
        train_types,
    )

    wandb_cfg = cfg.get("wandb", {})
    if OmegaConf.select(wandb_cfg, "enabled", default=False):
        try:
            import wandb

            wandb.init(
                project=OmegaConf.select(
                    wandb_cfg, "project", default="b2b-crossdomain"
                ),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"amorpheus_{'_'.join(t[:3] for t in train_types)}",
            )
        except ImportError:
            logger.warning("wandb not installed; skipping init")

    morph = _get_morphology(train_types[0], task)
    logger.info("Morphology: %d nodes", len(morph.nodes))

    policy = AmorpheusPolicy(
        morphology=morph,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    total_params = sum(p.numel() for p in policy.parameters())
    logger.info("Policy parameters: %d", total_params)

    resample_every = int(cfg.get("resample_every_n_epochs", 10))
    n_iterations = total_timesteps // rollout_steps
    steps_done = 0
    current_type_idx = 0
    train_env: gym.Env | None = None

    for iteration in range(n_iterations):
        if iteration % resample_every == 0 or train_env is None:
            if train_env is not None:
                train_env.close()
            bt = train_types[current_type_idx % len(train_types)]
            current_type_idx += 1
            train_env = _make_env(bt, iteration % 8, task)
            morph = train_env.metadata["morphology"]
            policy.morphology = morph
            logger.info("Resampled environment: %s", bt)

        rollout = collect_rollout(train_env, policy, rollout_steps)
        stats = ppo_update(policy, optimizer, rollout)
        steps_done += rollout_steps

        if iteration % 10 == 0:
            ep_reward = rollout["rewards"].sum().item()
            logger.info(
                "Iter %d/%d  steps=%d  loss=%.4f  ep_reward=%.1f",
                iteration,
                n_iterations,
                steps_done,
                stats["loss"],
                ep_reward,
            )

    if train_env is not None:
        train_env.close()

    torch.save(policy.state_dict(), output_dir / "amorpheus_policy.pt")
    logger.info("Saved policy to %s", output_dir / "amorpheus_policy.pt")

    logger.info("Evaluating zero-shot transfer...")
    for bt in train_types:
        for i in range(min(test_per_type, 3)):
            try:
                test_env = b2b.new_make_env(bt, split="test", index=i, task=task)
                test_morph = test_env.metadata["morphology"]
                policy.morphology = test_morph
                result = run_episode(test_env, policy)
                logger.info(
                    "Test %s[%d]: reward=%.1f  len=%d",
                    bt, i, result.total_reward, result.episode_length,
                )
                test_env.close()
            except Exception:
                logger.exception("Failed test eval: %s[%d]", bt, i)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
