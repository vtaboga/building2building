import numpy as np
import gymnasium as gym
import json
from pathlib import Path
from typing import Dict, Any
from building2building.core.run_manager import RunManager
from building2building.simulator.utils import TrajectoryLogger
from building2building.utils.results_parsing import parse_trajectories
from building2building.simulator.wrappers import CustomRescaleAction, NormalizeObservation
# Make sure to import your environment to register it
import building2building.simulator

def constant_policy(
    obs: np.ndarray,
    heating_setpoint: float = 21.0,
    cooling_setpoint: float = 24.0,
    normalize: bool = False,
    action_space: gym.spaces.Box = None
) -> np.ndarray:
    # The first value is the heating setpoint
    # The second value is the offset from the heating setpoint to the cooling setpoint
    if normalize:
        assert action_space is not None, "action_space must be provided when normalize=True"
        action = np.array([heating_setpoint, cooling_setpoint - heating_setpoint])
        # Normalize each action dimension to [0, 1]
        action = (action - action_space.low) / (action_space.high - action_space.low)
    else:
        assert cooling_setpoint > heating_setpoint
        action = np.array([heating_setpoint, cooling_setpoint - heating_setpoint])
    return action

def run_constant_baseline(
    env_id: str,
    path_to_building: str,
    path_to_weather: str,
    building_characteristics: Dict[str, Any],
    heating_setpoint: float = 21.0,
    cooling_setpoint: float = 24.0,
    reward_type: str = "base",
    energy_weight: float = 0.0,
    seed: int = 1,
) -> Dict[str, Any]:
    """
    Run a full year simulation using the constant policy baseline.
    
    Args:
        env_id: The gymnasium environment ID
        path_to_building: Path to the building epJSON file
        path_to_weather: Path to the weather file
        building_characteristics: Dictionary containing building characteristics
        heating_setpoint: Constant heating setpoint temperature
        cooling_setpoint: Constant cooling setpoint temperature
        reward_type: Type of reward function to use
        energy_weight: Weight of the energy consumption penalty
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing the evaluation results
    """
    # Create run manager for logging
    run_manager = RunManager(
        experiment_name="constant_baseline",
        track_wandb=False,
        seed=seed,
        tags={"policy": "constant"}
    )
    
    # Create and wrap the environment
    env = gym.make(
        env_id,
        path_to_building=path_to_building,
        path_to_weather=path_to_weather,
        building_characteristics=building_characteristics,
        reward_type=reward_type,
        energy_weight=energy_weight,
        run_manager=run_manager
    )
    action_space = env.action_space
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)

    uncontrolled_zones = env.unwrapped.uncontrolled_zones
    controlled_zones = env.unwrapped.controlled_zones
    observation_names = env.unwrapped.observation_names

    # Initialize trajectory logger
    trajectory_logger = TrajectoryLogger(
        run_manager.data_dir if run_manager else "trajectories",
        observation_names,
        logger=run_manager.logger
    )

    # Initialize metrics
    episode_reward = 0
    rewards = []
    timesteps = 0
    
    # Run the simulation
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    
    run_manager.logger.info(f"Starting constant baseline evaluation with heating={heating_setpoint}°C, cooling={cooling_setpoint}°C")
    run_manager.logger.info(f"Controlled zones: {controlled_zones}")
    run_manager.logger.info(f"Uncontrolled zones: {uncontrolled_zones}")
    
    while not (done or truncated):
        # Get action from constant policy
        action = constant_policy(obs, heating_setpoint, cooling_setpoint, normalize=True, action_space=action_space)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Log the trajectory
        trajectory_logger.log(env.denormalize(obs), env.env.env.scale_action(action), reward, controlled_zones, uncontrolled_zones)
        
        # Track metrics
        episode_reward += reward
        rewards.append(reward)
        timesteps += 1
        
        if timesteps % 24 == 0:  # Log every 24 timesteps (daily)
            run_manager.logger.debug(f"Day {timesteps//24}: Reward = {sum(rewards[-24:]):.2f}")
    
    # Calculate metrics
    results = {
        "total_reward": episode_reward,
        "mean_reward": episode_reward / timesteps,
        "std_reward": np.std(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "total_timesteps": timesteps,
        "heating_setpoint": heating_setpoint,
        "cooling_setpoint": cooling_setpoint
    }
    
    # Save results
    results_path = Path(run_manager.run_dir) / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Save the trajectories
    trajectory_logger.save()
    
    # Log final results
    run_manager.logger.info("Constant baseline evaluation completed")
    run_manager.logger.info(f"Total reward: {results['total_reward']:.2f}")
    run_manager.logger.info(f"Mean reward per step: {results['mean_reward']:.2f}")
    
    env.close()

    # Parse the trajectories
    parse_trajectories(trajectory_logger.trajectories_path)
    run_manager.logger.info("Results parsed")

    run_manager.finish()
    
    return results


