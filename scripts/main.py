import argparse
from datetime import datetime
from src.generator.search_idf import search_idf
from src.core.logging import setup_logger
from src.simulator.create_simulator import create_simulator
from src.generator.processing import add_hvac_meters_to_epjson, add_outdoor_air_meters_to_epjson, modify_timestep
import numpy as np


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
    
    # Create timestamp for unique log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"idf_generation_{timestamp}.log"
    
    # Setup root logger with the unique filename
    # This will be the parent logger for all other loggers
    root_logger = setup_logger('root', filename=log_filename)
    
    # Setup generator logger as a child of root, but don't add handlers
    # It will inherit handlers from root
    generator_logger = setup_logger('generator', filename=log_filename, add_handlers=False)
    
    # root_logger.info("Starting IDF generation process...")
    
    # Call search_idf with parsed arguments
    """
    buildings, path_to_weather = search_idf(
        state=args.state,
        county=args.county,
        building_type=args.building_type,
        area=args.area,
        num_floors=args.num_floors,
        height=args.height,
        n_buildings=args.n_buildings
    )"""

    root_logger.info("Create simulator...")

    # Process first building
    building = buildings[0]
    add_hvac_meters_to_epjson(building)
    add_outdoor_air_meters_to_epjson(building)
    modify_timestep(building)

    # Create gym environment
    env = create_simulator(
        path_to_building=building,
        path_to_weather=path_to_weather
    )

    root_logger.info("Starting simulation episodes...")

    # Run episodes
    for episode in range(args.episodes):
        root_logger.info(f"Episode {episode + 1}/{args.episodes}")
        
        # Reset environment
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Run episode
        while not done:
            # Simple action: constant temperature setpoints
            # You might want to implement a more sophisticated policy
            action = np.array([20.0, 25.0])  # Example: heating=20°C, cooling=25°C
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if step % 24 == 0:  # Log every 24 steps
                root_logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            step += 1
        
        root_logger.info(f"Episode {episode + 1} finished. Total steps: {step}, Total reward: {total_reward:.2f}")

    root_logger.info("Simulation complete.")
    env.close()
