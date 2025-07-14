"""
Script to collect offline datasets for offline RL training using Hydra configuration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import building2building.simulator
import gymnasium as gym
import hydra
import numpy as np
import torch
from building2building.algorithms.online.baselines import constant_policy
from building2building.env import DataPaths
from building2building.simulator.wrappers import (
    CustomRescaleAction,
    NormalizeObservation,
)
from building2building.types import BuildingCharacteristics, BuildingConfig
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.distributions import Normal


def load_dqn_policy(model_path: str, env: gym.Env, device: str = "cpu"):
    """Load trained DQN policy."""
    import gymnasium as gym
    from building2building.algorithms.online.dqn import QNetwork

    # Create a vectorized environment for the QNetwork initialization
    envs = gym.vector.SyncVectorEnv([lambda: env])

    policy = QNetwork(envs, bins_per_dimension=20).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    def get_action(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ = policy.get_action(obs_tensor, epsilon=0.0)  # No exploration
        return action

    return get_action


def load_ppo_policy(model_path: str, env: gym.Env, device: str = "cpu"):
    """Load trained PPO policy."""
    import gymnasium as gym
    from building2building.algorithms.online.ppo import Agent

    # Create a vectorized environment for the Agent initialization
    envs = gym.vector.SyncVectorEnv([lambda: env])

    policy = Agent(envs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    def get_action(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mean = policy.actor_mean(obs_tensor)
            action_logstd = policy.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        return action.cpu().numpy().flatten()

    return get_action


def collect_data_from_policy(
    env: gym.Env, policy_fn, num_episodes: int, logger: logging.Logger
) -> Dict[str, List]:
    """Collect data using any policy function in D4RL format."""

    data = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = policy_fn(obs)

            # Store current observation and action
            data["observations"].append(obs.copy())
            data["actions"].append(action.copy())

            # Take step
            next_obs, reward, done, truncated, info = env.step(action)

            # Store next observation and outcomes
            data["next_observations"].append(next_obs.copy())
            data["rewards"].append(reward)
            data["terminals"].append(done)
            data["timeouts"].append(truncated)

            obs = next_obs

        logger.info(f"Completed episode {episode + 1}/{num_episodes}")

    return data


def save_dataset(data: Dict[str, List], output_path: str, logger: logging.Logger):
    """Save dataset to file."""

    # Convert lists to numpy arrays
    dataset = {}
    for key, values in data.items():
        if key in ["rewards", "terminals", "timeouts"]:
            dataset[key] = np.array(values).reshape(-1, 1)
        else:
            dataset[key] = np.array(values)

    # Log dataset statistics
    n_transitions = len(dataset["observations"])
    total_reward = np.sum(dataset["rewards"])
    mean_reward = np.mean(dataset["rewards"])

    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total transitions: {n_transitions}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Mean reward per step: {mean_reward:.4f}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as NPZ format
    np.savez_compressed(output_path, **dataset)
    logger.info(f"Dataset saved to: {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="collect_dataset")
def main(cfg: DictConfig) -> None:
    """Main function for dataset collection with Hydra configuration."""

    # Get Hydra's output directory and setup logging
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger(__name__)

    # Load building characteristics
    building_path = (
        DataPaths.processed_dir() / f"{cfg.state}/{cfg.county}/{cfg.building_id}.epJSON"
    )
    characteristics_path = building_path.with_suffix(".json")

    weather_path = DataPaths.weather_dir() / cfg.weather_validation

    building_characteristics = BuildingCharacteristics.load_json(characteristics_path)

    building_config = BuildingConfig(
        building_path,
        weather_path,
        building_characteristics,
        cfg.reward_type,
        cfg.energy_weight,
        eplus_output_dir=output_dir,
    )

    # Create environment
    env = gym.make(
        "EnergyPlus-v0",
        building_config=building_config,
    )

    # Get action space before wrapping
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("Environment must have a Box action space")
    action_space = env.action_space

    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)

    # Set seed
    env.reset(seed=cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Determine policy type and load accordingly
    policy_type = cfg.dataset.get("policy_type", "constant")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if policy_type == "constant":
        logger.info(
            f"Starting constant data collection for {cfg.dataset.num_episodes} episodes"
        )
        policy_fn = lambda obs: constant_policy(
            obs,
            cfg.constant.heating_setpoint,
            cfg.constant.cooling_setpoint,
            normalize=True,
            action_space=action_space,
        )
        policy_name = "constant"

    elif policy_type == "dqn":
        model_path = cfg.dataset.model_path
        logger.info(f"Loading DQN policy from {model_path}")
        policy_fn = load_dqn_policy(model_path, env, device)
        policy_name = "dqn"

    elif policy_type == "ppo":
        model_path = cfg.dataset.model_path
        logger.info(f"Loading PPO policy from {model_path}")
        policy_fn = load_ppo_policy(model_path, env, device)
        policy_name = "ppo"

    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}. Use 'constant', 'dqn', or 'ppo'"
        )

    # Collect data
    logger.info(
        f"Starting {policy_type} data collection for {cfg.dataset.num_episodes} episodes"
    )
    data = collect_data_from_policy(env, policy_fn, cfg.dataset.num_episodes, logger)

    output_path = output_dir / f"{cfg.building_id}_{policy_name}_dataset.npz"
    save_dataset(data, str(output_path), logger)

    env.close()
    logger.info("Dataset collection completed!")


if __name__ == "__main__":
    main()
