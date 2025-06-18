"""
Offline RL training entry point for Building2Building environment.
"""

import os
import json
import argparse
import numpy as np
import torch
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional, Dict, Any

from building2building.algorithms.offline import MFPolicyTrainer, MBPolicyTrainer
from building2building.algorithms.offline.buffer import ReplayBuffer
from building2building.algorithms.offline.utils import Logger, load_dataset
from building2building.algorithms.offline.policy.model_free.cql import CQLPolicy
from building2building.algorithms.offline.policy.model_free.iql import IQLPolicy
from building2building.algorithms.offline.policy.model_free.td3bc import TD3BCPolicy
from building2building.algorithms.offline.modules import Actor, Critic
from building2building.algorithms.offline.nets import MLP
from building2building.core.run_manager import RunManager

import building2building.simulator


@dataclass
class OfflineRLArgs:
    """Arguments for offline RL training."""
    exp_name: str
    env_id: str
    path_to_building: str
    path_to_weather: str
    weather_validation: Optional[str]
    building_characteristics: Dict[str, Any]
    reward_type: str
    energy_weight: float
    dataset_path: str
    algorithm: str
    track: bool
    wandb_project_name: str
    wandb_entity: str
    results_dir: str
    seed: int
    save_model: bool
    epoch: int
    step_per_epoch: int
    eval_episodes: int
    batch_size: int
    lr: float
    gamma: float
    tau: float
    # CQL specific
    cql_weight: float = 1.0
    temperature: float = 1.0
    with_lagrange: bool = True
    lagrange_threshold: float = 10.0
    cql_alpha_lr: float = 1e-4
    # IQL specific
    beta: float = 3.0
    iql_tau: float = 0.7
    max_target_backup: bool = False
    # TD3+BC specific
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5


def create_policy(algorithm: str, obs_dim: int, action_dim: int, action_space, device: str, args: OfflineRLArgs):
    """Create policy based on algorithm."""
    actor = Actor(obs_dim, action_dim, device=device)
    
    if algorithm == "cql":
        critic1 = Critic(obs_dim, action_dim, device=device)
        critic2 = Critic(obs_dim, action_dim, device=device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
        
        policy = CQLPolicy(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            action_space=action_space,
            tau=args.tau,
            gamma=args.gamma,
            cql_weight=args.cql_weight,
            temperature=args.temperature,
            with_lagrange=args.with_lagrange,
            lagrange_threshold=args.lagrange_threshold,
            cql_alpha_lr=args.cql_alpha_lr
        )
    
    elif algorithm == "iql":
        critic_q1 = Critic(obs_dim, action_dim, device=device)
        critic_q2 = Critic(obs_dim, action_dim, device=device)
        critic_v = MLP([obs_dim, 256, 256, 1], device=device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=args.lr)
        critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=args.lr)
        critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.lr)
        
        policy = IQLPolicy(
            actor=actor,
            critic_q1=critic_q1,
            critic_q2=critic_q2,
            critic_v=critic_v,
            actor_optim=actor_optim,
            critic_q1_optim=critic_q1_optim,
            critic_q2_optim=critic_q2_optim,
            critic_v_optim=critic_v_optim,
            action_space=action_space,
            tau=args.tau,
            gamma=args.gamma,
            expectile=args.iql_tau,
            temperature=args.beta
        )
    
    elif algorithm == "td3bc":
        critic1 = Critic(obs_dim, action_dim, device=device)
        critic2 = Critic(obs_dim, action_dim, device=device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
        
        policy = TD3BCPolicy(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            update_actor_freq=args.policy_freq,
            alpha=args.alpha
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return policy


def main(args: OfflineRLArgs, run_manager: RunManager):
    """Main training function."""
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env_kwargs = {
        'path_to_building': args.path_to_building,
        'path_to_weather': args.path_to_weather,
        'building_characteristics': args.building_characteristics,
        'reward_type': args.reward_type,
        'energy_weight': args.energy_weight,
        'run_manager': run_manager
    }
    
    env = gym.make(args.env_id, **env_kwargs)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    buffer = ReplayBuffer(
        buffer_size=len(dataset['observations']),
        obs_shape=dataset['observations'].shape[1:],
        obs_dtype=np.float32,
        action_dim=dataset['actions'].shape[1],
        action_dtype=np.float32,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Add data to buffer
    buffer.add_batch(
        observations=dataset['observations'],
        actions=dataset['actions'],
        rewards=dataset['rewards'],
        next_observations=dataset['next_observations'],
        terminals=dataset['terminals']
    )
    
    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy = create_policy(args.algorithm, obs_dim, action_dim, env.action_space, device, args)
    
    # Create logger
    output_config = {"stdout": "stdout", "tensorboard": "tensorboard"}
    logger = Logger(args.results_dir, output_config)
    
    # Create trainer
    trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    )
    
    # Train
    run_manager.logger.info(f"Starting {args.algorithm.upper()} training...")
    results = trainer.train()
    run_manager.logger.info(f"Training completed. Final performance: {results}")
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.results_dir, f"{args.algorithm}_model.pth")
        torch.save(policy.state_dict(), model_path)
        run_manager.logger.info(f"Model saved to {model_path}")
    
    env.close()
    return results


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    parser = argparse.ArgumentParser()
    # Add arguments as needed for direct execution
    args = parser.parse_args()
    # Create dummy args and run_manager for testing
    # main(args, run_manager) 