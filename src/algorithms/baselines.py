import numpy as np
import gymnasium as gym
import json
from pathlib import Path
from typing import Dict, Any
from src.core.run_manager import RunManager

# Make sure to import your environment to register it
import src.simulator

def constant_policy(obs: np.ndarray, heating_setpoint: float = 21.0, cooling_setpoint: float = 24.0) -> np.ndarray:
    return np.array([cooling_setpoint, heating_setpoint])

def run_constant_baseline(
    env_id: str,
    path_to_building: str,
    path_to_weather: str,
    building_characteristics: Dict[str, Any],
    heating_setpoint: float = 21.0,
    cooling_setpoint: float = 24.0,
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
        run_manager=run_manager
    )
    
    # Set random seed
    env.reset(seed=seed)
    
    # Initialize metrics
    episode_reward = 0
    rewards = []
    timesteps = 0
    
    # Run the simulation
    obs, _ = env.reset()
    done = False
    truncated = False
    
    run_manager.logger.info(f"Starting constant baseline evaluation with heating={heating_setpoint}°C, cooling={cooling_setpoint}°C")
    
    while not (done or truncated):
        # Get action from constant policy
        action = constant_policy(obs, heating_setpoint, cooling_setpoint)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        
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
    
    # Log final results
    run_manager.logger.info("Constant baseline evaluation completed")
    run_manager.logger.info(f"Total reward: {results['total_reward']:.2f}")
    run_manager.logger.info(f"Mean reward per step: {results['mean_reward']:.2f}")
    
    env.close()
    run_manager.finish()
    
    return results


