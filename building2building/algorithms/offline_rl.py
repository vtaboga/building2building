"""
Offline RL algorithms adapted for Building2Building environment.
Integrates OfflineRL-Kit algorithms with the existing building simulation infrastructure.
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from pathlib import Path

# OfflineRL-Kit imports
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.policy import CQLPolicy, IQLPolicy, TD3BCPolicy
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.utils.logger import Logger, make_log_dirs

# Building2Building imports
from building2building.simulator.wrappers import NormalizeObservation, CustomRescaleAction
from building2building.core.run_manager import RunManager
import building2building.simulator

# Set default tensor type to float64 for better precision (matching your existing code)
torch.set_default_dtype(torch.float64)


@dataclass 
class OfflineRLArgs:
    """Arguments for offline RL training in Building2Building environment."""
    
    # Experiment configuration
    exp_name: str = "offline_rl"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "building2building-offline"
    wandb_entity: str = None
    save_model: bool = False
    results_dir: str = "results"
    
    # Environment configuration
    env_id: str = "EnergyPlus-v0"
    path_to_building: str = None
    path_to_weather: str = None 
    weather_validation: str = None
    building_characteristics: dict = None
    reward_type: str = "base"
    energy_weight: float = 1.0
    
    # Dataset configuration
    dataset_path: str = None  # Path to offline dataset
    buffer_size: int = 1000000
    
    # Algorithm configuration
    algorithm: str = "cql"  # Options: "cql", "iql", "td3bc"
    
    # Training configuration
    epoch: int = 1000
    step_per_epoch: int = 1000
    eval_episodes: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    
    # Algorithm-specific parameters
    # Common
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dims: tuple = (256, 256)
    
    # CQL specific
    cql_weight: float = 1.0
    temperature: float = 1.0
    with_lagrange: bool = False
    lagrange_threshold: float = 10.0
    cql_alpha_lr: float = 3e-4
    
    # IQL specific
    beta: float = 3.0
    iql_tau: float = 0.7
    max_target_backup: bool = False
    
    # TD3+BC specific
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5


def load_building_dataset(dataset_path: str, env: gym.Env) -> Dict[str, np.ndarray]:
    """
    Load offline dataset for building environment.
    
    Args:
        dataset_path: Path to the dataset (could be JSON, NPZ, or custom format)
        env: Environment instance for validation
    
    Returns:
        Dictionary with keys: observations, actions, rewards, terminals, timeouts
    """
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Convert to numpy arrays and validate shapes
        dataset = {
            'observations': np.array(data['observations'], dtype=np.float32),
            'actions': np.array(data['actions'], dtype=np.float32),
            'rewards': np.array(data['rewards'], dtype=np.float32),
            'terminals': np.array(data['terminals'], dtype=bool),
            'timeouts': np.array(data.get('timeouts', [False] * len(data['rewards'])), dtype=bool)
        }
        
    elif dataset_path.endswith('.npz'):
        data = np.load(dataset_path)
        dataset = {
            'observations': data['observations'].astype(np.float32),
            'actions': data['actions'].astype(np.float32), 
            'rewards': data['rewards'].astype(np.float32),
            'terminals': data['terminals'].astype(bool),
            'timeouts': data.get('timeouts', np.zeros_like(data['terminals'], dtype=bool))
        }
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    # Validate dataset
    n = len(dataset['observations'])
    for key, value in dataset.items():
        assert len(value) == n, f"Dataset component {key} has length {len(value)}, expected {n}"
    
    # Validate shapes match environment
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    assert dataset['observations'].shape[1:] == obs_shape, \
        f"Observation shape mismatch: dataset {dataset['observations'].shape[1:]}, env {obs_shape}"
    assert dataset['actions'].shape[1:] == action_shape, \
        f"Action shape mismatch: dataset {dataset['actions'].shape[1:]}, env {action_shape}"
    
    print(f"Loaded dataset with {n} transitions")
    return dataset


def main(args: OfflineRLArgs, run_manager: Optional[RunManager] = None):
    """Main function for offline RL training."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create environment
    env = gym.make(
        args.env_id,
        path_to_building=args.path_to_building,
        path_to_weather=args.path_to_weather,
        building_characteristics=args.building_characteristics,
        reward_type=args.reward_type,
        energy_weight=args.energy_weight,
        run_manager=run_manager
    )
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    
    # Load offline dataset
    dataset = load_building_dataset(args.dataset_path, env)
    
    # Create networks and policy based on algorithm
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    
    actor_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=args.hidden_dims)
    
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    
    actor = ActorProb(actor_backbone, dist, device)
    critic1 = Critic(critic1_backbone, device)
    critic2 = Critic(critic2_backbone, device)
    
    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
    
    # Create policy based on algorithm choice
    if args.algorithm.lower() == "cql":
        policy = CQLPolicy(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            action_space=env.action_space,
            tau=args.tau,
            gamma=args.gamma,
            alpha=None,  # Will be auto-tuned
            cql_weight=args.cql_weight,
            temperature=args.temperature,
            max_q_backup=False,
            deterministic_backup=True,
            with_lagrange=args.with_lagrange,
            lagrange_threshold=args.lagrange_threshold,
            cql_alpha_lr=args.cql_alpha_lr,
            num_repeart_actions=10
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    # Create replay buffer and load dataset
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=env.observation_space.shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device=device
    )
    buffer.load_dataset(dataset)
    
    # Setup logging
    if run_manager:
        logger = run_manager.logger
    else:
        log_dirs = make_log_dirs(args.env_id, args.algorithm, args.seed, vars(args))
        output_config = {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "tb": "tensorboard"
        }
        logger = Logger(log_dirs, output_config)
        logger.log_hyperparameters(vars(args))
    
    # Create trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    )
    
    # Train policy
    logger.info(f"Starting {args.algorithm.upper()} training...")
    policy_trainer.train()
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.results_dir, "model")
        os.makedirs(model_path, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(model_path, "policy.pth"))
        logger.info(f"Model saved to {model_path}")
    
    logger.info(f"{args.algorithm.upper()} training completed!")
    env.close()


def offline_rl_evaluate(
    model_path: str,
    args: OfflineRLArgs,
    eval_episodes: int = 10,
    run_manager: Optional[RunManager] = None,
    save_trajectories: bool = False
):
    """Evaluate trained offline RL policy."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create environment
    env_fn = gym.make(
        args.env_id,
        path_to_building=args.path_to_building,
        path_to_weather=args.path_to_weather,
        building_characteristics=args.building_characteristics,
        reward_type=args.reward_type,
        energy_weight=args.energy_weight,
        run_manager=run_manager
    )
    env = env_fn()
    
    # Create and load policy
    policy = create_offline_rl_policy(args, env, device)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            with torch.no_grad():
                action = policy.select_action(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if run_manager:
            run_manager.logger.info(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Calculate statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episodes": eval_episodes
    }
    
    if run_manager:
        run_manager.logger.info(f"Evaluation Results: {results}")
        
        # Save results
        results_path = Path(run_manager.run_dir) / "offline_rl_eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    
    env.close()
    return results 