import numpy as np
import gymnasium as gym
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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
    action_space: Optional[gym.spaces.Box] = None
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
    results_dir: str = "baseline_results",
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
        results_dir: Directory to save results
    
    Returns:
        Dictionary containing the evaluation results
    """
    # Setup logging and results directory
    os.makedirs(results_dir, exist_ok=True)
    logger = logging.getLogger("baseline")
    
    # Create and wrap the environment
    env = gym.make(
        env_id,
        path_to_building=path_to_building,
        path_to_weather=path_to_weather,
        building_characteristics=building_characteristics,
        reward_type=reward_type,
        energy_weight=energy_weight
    )
    
    # Get action space before wrapping and ensure it's a Box space
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("Environment must have a Box action space for the constant policy")
    action_space = env.action_space
    
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)

    # Get environment properties safely
    base_env = env.unwrapped
    uncontrolled_zones = getattr(base_env, 'uncontrolled_zones', None)
    controlled_zones = getattr(base_env, 'controlled_zones', None)
    observation_names = getattr(base_env, 'observation_names', None)

    # Initialize trajectory logger
    trajectory_logger = TrajectoryLogger(
        os.path.join(results_dir, "trajectories"),
        observation_names,
        logger=logger
    )

    # Initialize metrics
    episode_reward = 0.0
    rewards = []
    timesteps = 0
    
    # Run the simulation
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    
    logger.info(f"Starting constant baseline evaluation with heating={heating_setpoint}°C, cooling={cooling_setpoint}°C")
    logger.info(f"Controlled zones: {controlled_zones}")
    logger.info(f"Uncontrolled zones: {uncontrolled_zones}")
    
    while not (done or truncated):
        # Get action from constant policy
        action = constant_policy(obs, heating_setpoint, cooling_setpoint, normalize=True, action_space=action_space)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Find the wrappers to get denormalized observation and scaled action
        norm_wrapper = None
        rescale_wrapper = None
        temp_env = env

        while temp_env is not None:
            if isinstance(temp_env, NormalizeObservation):
                norm_wrapper = temp_env
            if isinstance(temp_env, CustomRescaleAction):
                rescale_wrapper = temp_env
            
            if hasattr(temp_env, 'env'):
                temp_env = getattr(temp_env, 'env')
            else:
                break
        
        # Log the trajectory with proper denormalization and scaling
        if norm_wrapper and rescale_wrapper:
            denorm_obs = norm_wrapper.denormalize(obs)
            scaled_action = rescale_wrapper.scale_action(action)
            trajectory_logger.log(denorm_obs, scaled_action, reward, controlled_zones, uncontrolled_zones)
        else:
            # Fallback if wrappers not found
            trajectory_logger.log(obs, action, reward, controlled_zones, uncontrolled_zones)
        
        # Track metrics
        episode_reward += float(reward)
        rewards.append(float(reward))
        timesteps += 1
        
        if timesteps % 24 == 0:  # Log every 24 timesteps (daily)
            logger.debug(f"Day {timesteps//24}: Reward = {sum(rewards[-24:]):.2f}")
    
    # Calculate metrics
    results = {
        "total_reward": float(episode_reward),
        "mean_reward": float(episode_reward / timesteps),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(min(rewards)),
        "max_reward": float(max(rewards)),
        "total_timesteps": timesteps,
        "heating_setpoint": heating_setpoint,
        "cooling_setpoint": cooling_setpoint
    }
    
    # Save results
    results_path = Path(results_dir) / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Save the trajectories
    trajectory_logger.save()
    
    # Log final results
    logger.info("Constant baseline evaluation completed")
    logger.info(f"Total reward: {results['total_reward']:.2f}")
    logger.info(f"Mean reward per step: {results['mean_reward']:.2f}")
    
    env.close()

    # Parse the trajectories
    parse_trajectories(trajectory_logger.trajectories_path)
    logger.info("Results parsed")
    
    return results


