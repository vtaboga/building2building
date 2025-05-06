import argparse
from datetime import datetime
import src.simulator.utils
from src.generator.search_idf import search_idf
from src.core.run_manager import RunManager
from src.generator.processing import add_hvac_meters_to_epjson, add_outdoor_air_meters_to_epjson, modify_timestep, add_setpoint_control_to_epjson
import numpy as np
import os
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate IDF files based on specified parameters')
    
    parser.add_argument('--state', type=str, default="AL",
                       help='State code (e.g., AL)')
    parser.add_argument('--county', type=str, default="Pike",
                       help='County name')
    parser.add_argument('--building-type', type=str, default="SmallHotel",
                       help='Building type (e.g., SmallHotel)')
    parser.add_argument('--area', type=float, default=1140,
                       help='Building area in square feet')
    parser.add_argument('--num-floors', type=int, default=6,
                       help='Number of floors')
    parser.add_argument('--height', type=float, default=None,
                       help='Height of the building')
    parser.add_argument('--n-buildings', type=int, default=1,
                       help='Number of buildings to search for')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--track', action='store_true',
                       help='Track with wandb')
    parser.add_argument('--wandb-project', type=str, default="building2building",
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='W&B entity')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create a run manager for this experiment
    run_manager = RunManager(
        experiment_name="simulation",
        track_wandb=args.track,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        tags={
            "state": args.state,
            "county": args.county,
            "building_type": args.building_type,
            "area": args.area,
            "num_floors": args.num_floors,
            "height": args.height,
            "n_buildings": args.n_buildings,
            "episodes": args.episodes
        }
    )
    
    # Get logger from run manager
    logger = run_manager.logger
    
    logger.info("Searching for building files...")
    buildings, path_to_weather = search_idf(
        state=args.state,
        county=args.county,
        building_type=args.building_type,
        area=args.area,
        num_floors=args.num_floors,
        height=args.height,
        n_buildings=args.n_buildings
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

    # Run episodes
    for episode in range(args.episodes):
        logger.info(f"Episode {episode + 1}/{args.episodes}")
        
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
                if args.track:
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
        'total_episodes': args.episodes,
        'average_reward': np.mean([ep['total_reward'] for ep in episode_results]),
        'average_length': np.mean([ep['episode_length'] for ep in episode_results]),
    }
    
    run_manager.save_results(summary, 'simulation_summary.json')
    logger.info("Simulation complete. Results saved in: " + run_manager.run_dir)
    env.close()
    
    # Finalize the run
    run_manager.finish()
