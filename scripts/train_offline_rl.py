"""
Training script for offline RL algorithms using Hydra configuration.
"""

import os
import json
import hydra
import logging
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# Import wandb for tracking
import wandb

# Import the necessary components directly
from building2building.algorithms.offline import MFPolicyTrainer, MBPolicyTrainer
from building2building.algorithms.offline.buffer import ReplayBuffer
from building2building.algorithms.offline.utils import Logger, load_dataset
from building2building.algorithms.offline.policy.model_free.cql import CQLPolicy
from building2building.algorithms.offline.policy.model_free.iql import IQLPolicy
from building2building.algorithms.offline.policy.model_free.td3bc import TD3BCPolicy
from building2building.algorithms.offline.modules import Actor, Critic, ActorProb, TanhDiagGaussian
from building2building.algorithms.offline.nets import MLP
from building2building.simulator.wrappers import CustomRescaleAction, NormalizeObservation
import building2building.simulator


def create_policy(algorithm: str, obs_dim: int, action_dim: int, action_space, device: str, cfg: DictConfig):
    """Create policy based on algorithm."""
    
    if algorithm == "cql":
        # Create MLP backbone for actor (processes observations)
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        # Use ActorProb for CQL (needs distributions)
        dist_net = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True
        ).to(device)
        actor = ActorProb(actor_backbone, dist_net, device=device)
        
        # Create MLP backbones for critics (process obs+action)
        critic1_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic2_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic1 = Critic(critic1_backbone, device=device)
        critic2 = Critic(critic2_backbone, device=device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.offline_rl.lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=cfg.offline_rl.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=cfg.offline_rl.lr)
        
        policy = CQLPolicy(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            action_space=action_space,
            tau=cfg.offline_rl.tau,
            gamma=cfg.offline_rl.gamma,
            cql_weight=getattr(cfg.offline_rl, 'cql_weight', 1.0),
            temperature=getattr(cfg.offline_rl, 'temperature', 1.0),
            with_lagrange=getattr(cfg.offline_rl, 'with_lagrange', True),
            lagrange_threshold=getattr(cfg.offline_rl, 'lagrange_threshold', 10.0),
            cql_alpha_lr=getattr(cfg.offline_rl, 'cql_alpha_lr', 1e-4)
        )
    
    elif algorithm == "iql":
        # Create MLP backbone for actor (processes observations)
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        # Use ActorProb for IQL (needs distributions)
        dist_net = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True
        ).to(device)
        actor = ActorProb(actor_backbone, dist_net, device=device)
        
        # Create MLP backbones for critics
        critic_q1_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic_q2_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic_q1 = Critic(critic_q1_backbone, device=device)
        critic_q2 = Critic(critic_q2_backbone, device=device)
        critic_v = MLP(obs_dim, [256, 256], 1).to(device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.offline_rl.lr)
        critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=cfg.offline_rl.lr)
        critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=cfg.offline_rl.lr)
        critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=cfg.offline_rl.lr)
        
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
            tau=cfg.offline_rl.tau,
            gamma=cfg.offline_rl.gamma,
            expectile=getattr(cfg.offline_rl, 'iql_tau', 0.7),
            temperature=getattr(cfg.offline_rl, 'beta', 3.0)
        )
    
    elif algorithm == "td3bc":
        # Create MLP backbone for actor (processes observations)
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        # Use regular Actor for TD3BC (deterministic actions)
        actor = Actor(actor_backbone, action_dim, device=device)
        
        # Create MLP backbones for critics
        critic1_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic2_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic1 = Critic(critic1_backbone, device=device)
        critic2 = Critic(critic2_backbone, device=device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.offline_rl.lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=cfg.offline_rl.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=cfg.offline_rl.lr)
        
        policy = TD3BCPolicy(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            tau=cfg.offline_rl.tau,
            gamma=cfg.offline_rl.gamma,
            policy_noise=getattr(cfg.offline_rl, 'policy_noise', 0.2),
            noise_clip=getattr(cfg.offline_rl, 'noise_clip', 0.5),
            update_actor_freq=getattr(cfg.offline_rl, 'policy_freq', 2),
            alpha=getattr(cfg.offline_rl, 'alpha', 2.5)
        )
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return policy


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for offline RL training with Hydra configuration."""
    
    # Get Hydra's output directory and setup logging
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger(__name__)
    
    # Initialize wandb if tracking is enabled
    wandb_run = None
    if cfg.get('track', False):
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=f"{cfg.offline_rl.algorithm}_{cfg.building.building_id}",
            tags=cfg.wandb.get('tags', [])
        )
        logger.info(f"W&B initialized: {wandb_run.url}")
    
    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}.json"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        logger.error(f"Building characteristics file not found: {characteristics_path}")
        raise
    
    # Validate dataset exists
    dataset_path = cfg.offline_rl.dataset_path
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Create environment (without run_manager dependency)
    env_kwargs = {
        'path_to_building': building_path,
        'path_to_weather': f"data/weather/{cfg.building.weather}",
        'building_characteristics': building_characteristics,
        'reward_type': cfg.reward_type,
        'energy_weight': cfg.energy_weight,
    }
    
    env = gym.make("EnergyPlus-v0", **env_kwargs)
    # Store original action space before wrappers modify it
    original_action_space = env.action_space
    
    # Apply the same wrappers as in dataset collection
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device_str}")
    
    # Create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset['observations']),
        obs_shape=dataset['observations'].shape[1:],
        obs_dtype=dataset['observations'].dtype,
        action_dim=dataset['actions'].shape[1],
        action_dtype=dataset['actions'].dtype,
        device=device_str
    )
    
    # Add data to buffer
    buffer.add_batch(
        obss=dataset['observations'],
        next_obss=dataset['next_observations'],
        actions=dataset['actions'],
        rewards=dataset['rewards'],
        terminals=dataset['terminals']
    )
    
    # Create policy - get dimensions from dataset
    obs_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    
    logger.info(f"Dataset loaded: {len(dataset['observations'])} transitions")
    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Use original action space for policy (before wrappers)
    policy = create_policy(cfg.offline_rl.algorithm, obs_dim, action_dim, original_action_space, device_str, cfg)
    
    # Create logger for training
    output_config = {"stdout": "stdout", "tensorboard": "tensorboard"}
    training_logger = Logger(str(output_dir), output_config)
    
    # Create trainer (only model-free algorithms for now)
    trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=training_logger,
        epoch=cfg.offline_rl.epoch,
        step_per_epoch=cfg.offline_rl.step_per_epoch,
        batch_size=cfg.offline_rl.batch_size,
        eval_episodes=cfg.offline_rl.eval_episodes
    )
    
    # Log the training start
    logger.info(f"Starting {cfg.offline_rl.algorithm.upper()} offline RL training...")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Building: {cfg.building.state}/{cfg.building.county}/{cfg.building.building_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Algorithm: {cfg.offline_rl.algorithm}")
    logger.info(f"Epochs: {cfg.offline_rl.epoch}")
    logger.info(f"Batch size: {cfg.offline_rl.batch_size}")
    logger.info(f"Learning rate: {cfg.offline_rl.lr}")
    
    # Train
    try:
        results = trainer.train()
        logger.info(f"{cfg.offline_rl.algorithm.upper()} training completed successfully")
        logger.info(f"Final performance: {results}")
        
        # Log final results to wandb if tracking
        if wandb_run:
            wandb.log({"final_performance": results})
        
        # Save model
        model_path = output_dir / f"{cfg.offline_rl.algorithm}_model.pth"
        torch.save(policy.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error during {cfg.offline_rl.algorithm.upper()} training: {e}")
        raise
    finally:
        env.close()
        # Finish wandb run
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    main_hydra() 