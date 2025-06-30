import hydra
from omegaconf import DictConfig
import src.simulator.utils
from building2building.generator.search_idf import search_idf
from building2building.core.hydra_manager import HydraManager
from building2building.generator.processing import add_hvac_meters_to_epjson, add_outdoor_air_meters_to_epjson, modify_timestep, add_setpoint_control_to_epjson
import numpy as np
import os
import json


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for running building simulation with Hydra configuration."""
    
    # Create a Hydra-compatible run manager
    run_manager = HydraManager(cfg)
    
    # Get logger from run manager
    logger = run_manager.logger
    
    # Extract building configuration
    building_cfg = cfg.building
    simulation_cfg = cfg.simulation
    
    logger.info("Searching for building files...")
    buildings, path_to_weather = search_idf(
        state=building_cfg.state,
        county=building_cfg.county,
        building_type=building_cfg.building_type,
        area=building_cfg.area,
        num_floors=building_cfg.num_floors,
        height=building_cfg.height,
        n_buildings=building_cfg.n_buildings
    )

    logger.info("Create simulator...")

    # Process first building
    building, characteristics = buildings[0]
    add_hvac_meters_to_epjson(building)
    add_outdoor_air_meters_to_epjson(building)
    modify_timestep(building)
    add_setpoint_control_to_epjson(building)

    # Import gymnasium and create registered environment
    import gymnasium as gym
    env_kwargs = {
        'path_to_building': building,
        'path_to_weather': path_to_weather,
        'building_characteristics': characteristics,
        'run_manager': run_manager  # Pass the run manager to the environment
    }
    env = gym.make('EnergyPlus-v0', **env_kwargs)

    logger.info("Starting simulation episodes...")
    
    # Create lists to store episode results
    episode_results = []
    episodes = simulation_cfg.episodes

    # Run episodes
    for episode in range(episodes):
        logger.info(f"Episode {episode + 1}/{episodes}")
        
        episode_data = {
            'steps': [],
            'rewards': [],
            'total_reward': 0,
            'episode_length': 0
        }
        
        # Reset environment
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Run episode
        for _ in range(10):
            action = np.array([20.0, 25.0]) 
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store step data
            episode_data['steps'].append({
                'step': step,
                'reward': float(reward),
                'observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
                'info': info
            })
            
            if step % 24 == 0:  # Log every 24 steps
                logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                
                # Log metrics to wandb if enabled
                if cfg.track:
                    run_manager.log_metrics({
                        'reward': float(reward),
                        'total_reward': float(total_reward)
                    }, step=step)
                    
            step += 1
        
        # Store episode summary
        episode_data['total_reward'] = float(total_reward)
        episode_data['episode_length'] = step
        episode_results.append(episode_data)
        
        logger.info(f"Episode {episode + 1} finished. Total steps: {step}, Total reward: {total_reward:.2f}")
        
        # Save episode results using run manager
        run_manager.save_results(episode_data, f'episode_{episode+1}_results.json')

    # Save summary of all episodes
    summary = {
        'total_episodes': episodes,
        'average_reward': np.mean([ep['total_reward'] for ep in episode_results]),
        'average_length': np.mean([ep['episode_length'] for ep in episode_results]),
    }
    
    run_manager.save_results(summary, 'simulation_summary.json')
    logger.info("Simulation complete. Results saved in: " + run_manager.run_dir)
    env.close()
    
    # Finalize the run
    run_manager.finish()


if __name__ == "__main__":
    main()
