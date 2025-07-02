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
from building2building.algorithms.offline.policy.model_based.mopo import MOPOPolicy
from building2building.algorithms.offline.policy.model_based.combo import COMBOPolicy
from building2building.algorithms.offline.policy.model_based.mobile import MOBILEPolicy
from building2building.algorithms.offline.policy.model_based.rambo import RAMBOPolicy
from building2building.algorithms.offline.dynamics import EnsembleDynamics, RNNDynamics
from building2building.algorithms.offline.modules import Actor, Critic
from building2building.algorithms.offline.nets import MLP
from building2building.core.run_manager import RunManager
from building2building.simulator.wrappers import CustomRescaleAction, NormalizeObservation

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
    # Model-based specific
    rollout_freq: int = 1000
    rollout_batch_size: int = 50000
    rollout_length: int = 5
    real_ratio: float = 0.05
    dynamics_lr: float = 1e-3
    dynamics_hidden_dims: tuple = (200, 200, 200, 200)
    dynamics_update_freq: int = 0
    n_ensemble: int = 7
    n_elites: int = 5


def create_policy(algorithm: str, obs_dim: int, action_dim: int, action_space, device: str, args: OfflineRLArgs):
    """Create policy based on algorithm."""
    
    if algorithm == "cql":
        # Create MLP backbone for actor (processes observations)
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        actor = Actor(actor_backbone, action_dim, device=device)
        
        # Create MLP backbones for critics (process obs+action)
        critic1_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic2_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic1 = Critic(critic1_backbone, device=device)
        critic2 = Critic(critic2_backbone, device=device)
        
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
        # Create MLP backbone for actor (processes observations)
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        actor = Actor(actor_backbone, action_dim, device=device)
        
        # Create MLP backbones for critics
        critic_q1_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic_q2_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic_q1 = Critic(critic_q1_backbone, device=device)
        critic_q2 = Critic(critic_q2_backbone, device=device)
        critic_v = MLP(obs_dim, [256, 256], 1).to(device)
        
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
        # Create MLP backbone for actor (processes observations)
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        actor = Actor(actor_backbone, action_dim, device=device)
        
        # Create MLP backbones for critics
        critic1_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic2_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic1 = Critic(critic1_backbone, device=device)
        critic2 = Critic(critic2_backbone, device=device)
        
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
    
    elif algorithm == "mopo":
        # Create ensemble model for dynamics
        from building2building.algorithms.offline.nets import EnsembleNetwork
        from building2building.algorithms.offline.utils.scaler import StandardScaler
        from building2building.algorithms.offline.modules import ActorProb, TanhDiagGaussian
        
        # Simple terminal function that always returns False (no early termination)
        def terminal_fn(obs, action, next_obs):
            return np.zeros((obs.shape[0], 1), dtype=bool)
        
        # Create ensemble model
        ensemble_model = EnsembleNetwork(
            input_dim=obs_dim + action_dim,
            output_dim=obs_dim + 1,  # next_obs + reward
            hidden_dims=args.dynamics_hidden_dims,
            num_ensemble=args.n_ensemble,
            num_elites=args.n_elites,
            device=device
        )
        
        # Create dynamics optimizer and scaler
        dynamics_optim = torch.optim.Adam(ensemble_model.parameters(), lr=args.dynamics_lr)
        scaler = StandardScaler()
        
        # Create dynamics model
        dynamics = EnsembleDynamics(
            model=ensemble_model,
            optim=dynamics_optim,
            scaler=scaler,
            terminal_fn=terminal_fn
        )
        
        # Create probabilistic actor for SAC-based MOPO
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        dist_net = TanhDiagGaussian(
            latent_dim=actor_backbone.output_dim,
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
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
        
        policy = MOPOPolicy(
            dynamics=dynamics,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            alpha=0.2
        )
    
    elif algorithm in ["combo", "mobile", "rambo"]:
        # For now, redirect these to MOPO until properly implemented
        import warnings
        warnings.warn(f"{algorithm} not fully implemented, using MOPO instead")
        # Create the same setup as MOPO...
        from building2building.algorithms.offline.nets import EnsembleNetwork
        from building2building.algorithms.offline.utils.scaler import StandardScaler
        from building2building.algorithms.offline.modules import ActorProb, TanhDiagGaussian
        
        def terminal_fn(obs, action, next_obs):
            return np.zeros((obs.shape[0], 1), dtype=bool)
        
        ensemble_model = EnsembleNetwork(
            input_dim=obs_dim + action_dim,
            output_dim=obs_dim + 1,
            hidden_dims=args.dynamics_hidden_dims,
            num_ensemble=args.n_ensemble,
            num_elites=args.n_elites,
            device=device
        )
        
        dynamics_optim = torch.optim.Adam(ensemble_model.parameters(), lr=args.dynamics_lr)
        scaler = StandardScaler()
        
        dynamics = EnsembleDynamics(
            model=ensemble_model,
            optim=dynamics_optim,
            scaler=scaler,
            terminal_fn=terminal_fn
        )
        
        # Create probabilistic actor for SAC-based policies
        actor_backbone = MLP(obs_dim, [256, 256]).to(device)
        dist_net = TanhDiagGaussian(
            latent_dim=actor_backbone.output_dim,
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True
        ).to(device)
        actor = ActorProb(actor_backbone, dist_net, device=device)
        
        # Create MLP backbones for critics
        critic1_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic2_backbone = MLP(obs_dim + action_dim, [256, 256]).to(device)
        critic1 = Critic(critic1_backbone, device=device)
        critic2 = Critic(critic2_backbone, device=device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
        
        policy = MOPOPolicy(
            dynamics=dynamics,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            alpha=0.2
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
    # Apply the same wrappers as in dataset collection
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    # Create policy - get dimensions from dataset instead of environment
    obs_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    
    policy = create_policy(args.algorithm, obs_dim, action_dim, env.action_space, device_str, args)
    
    # Initialize dynamics model for model-based algorithms
    if args.algorithm in ["mopo", "combo", "mobile", "rambo"]:
        print("Initializing dynamics model...")
        dynamics_info = policy.update_dynamics(buffer)
        print(f"Dynamics initialization completed. Loss: {dynamics_info.get('dynamics_loss', 'N/A')}")
    
    # Create logger with TensorBoard output
    output_config = {"stdout": "stdout", "tensorboard": "tensorboard"}
    logger = Logger(args.results_dir, output_config)
    
    # Determine if this is a model-based algorithm
    model_based_algorithms = ["mopo", "combo", "mobile", "rambo"]
    is_model_based = args.algorithm in model_based_algorithms
    
    if is_model_based:
        # Create fake buffer for model-based training
        fake_buffer = ReplayBuffer(
            buffer_size=args.rollout_batch_size * args.rollout_length,
            obs_shape=dataset['observations'].shape[1:],
            obs_dtype=dataset['observations'].dtype,
            action_dim=dataset['actions'].shape[1],
            action_dtype=dataset['actions'].dtype,
            device=device_str
        )
        
        # Use MBPolicyTrainer for model-based algorithms
        trainer = MBPolicyTrainer(
            policy=policy,
            eval_env=env,
            real_buffer=buffer,
            fake_buffer=fake_buffer,
            logger=logger,
            rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
            epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            batch_size=args.batch_size,
            real_ratio=args.real_ratio,
            eval_episodes=args.eval_episodes,
            dynamics_update_freq=args.dynamics_update_freq
        )
    else:
        # Use MFPolicyTrainer for model-free algorithms
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