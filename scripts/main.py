import argparse
from datetime import datetime
import src.simulator.utils
from src.generator.search_idf import search_idf
from src.core.logging import setup_logger
from src.simulator.create_simulator import create_simulator
from src.generator.processing import add_hvac_meters_to_epjson, add_outdoor_air_meters_to_epjson, modify_timestep
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
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create timestamp for unique folder and log file names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', f'simulation_{timestamp}')
    logs_dir = os.path.join(base_dir, 'logs')
    
    # Create all necessary directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save simulation parameters
    params = vars(args)
    with open(os.path.join(results_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Setup loggers with new path
    log_filename = os.path.join(results_dir, 'simulation.log')
    root_logger = setup_logger('root', filename=log_filename)
    generator_logger = setup_logger('generator', filename=log_filename, add_handlers=False)
    
    
    buildings, path_to_weather = search_idf(
        state=args.state,
        county=args.county,
        building_type=args.building_type,
        area=args.area,
        num_floors=args.num_floors,
        height=args.height,
        n_buildings=args.n_buildings
    )

    root_logger.info("Create simulator...")

    # Process first building
    building = buildings[0]
    add_hvac_meters_to_epjson(building)
    add_outdoor_air_meters_to_epjson(building)
    modify_timestep(building)

    # Import gymnasium and create registered environment
    import gymnasium as gym
    env_kwargs = {
        'path_to_building': building,
        'path_to_weather': path_to_weather
    }
    env = gym.make('EnergyPlus-v0', **env_kwargs)

    root_logger.info("Starting simulation episodes...")
    
    # Create lists to store episode results
    episode_results = []

    # Run episodes
    for episode in range(args.episodes):
        root_logger.info(f"Episode {episode + 1}/{args.episodes}")
        
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
                root_logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            step += 1
        
        # Store episode summary
        episode_data['total_reward'] = float(total_reward)
        episode_data['episode_length'] = step
        episode_results.append(episode_data)
        
        root_logger.info(f"Episode {episode + 1} finished. Total steps: {step}, Total reward: {total_reward:.2f}")
        
        # Save episode results
        with open(os.path.join(results_dir, f'episode_{episode+1}_results.json'), 'w') as f:
            json.dump(episode_data, f, indent=4)

    # Save summary of all episodes
    summary = {
        'total_episodes': args.episodes,
        'average_reward': np.mean([ep['total_reward'] for ep in episode_results]),
        'average_length': np.mean([ep['episode_length'] for ep in episode_results]),
        'timestamp': timestamp
    }
    
    with open(os.path.join(results_dir, 'simulation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    root_logger.info("Simulation complete. Results saved in: " + results_dir)
    env.close()
