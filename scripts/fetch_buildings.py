import argparse
from datetime import datetime
from src.generator.search_idf import search_idf
from src.core.run_manager import RunManager


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate IDF files based on specified parameters')
    
    parser.add_argument('--state', '-s', type=str, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, help='County name')
    parser.add_argument('--building-type', '-b', type=str, default="SmallOffice", help='Building type (e.g., SmallOffice)')
    parser.add_argument('--area', '-a', type=float, default=1000, help='Building area in square feet')
    parser.add_argument('--num-floors', '-f', type=int, default=1, help='Number of floors')
    parser.add_argument('--height', type=float, default=None, help='Height of the building')
    parser.add_argument('--n-buildings', '-n', type=int, default=1, help='Number of buildings to search for')

    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create a run manager for this experiment
    run_manager = RunManager(
        experiment_name="fetch_buildings",       
        tags={
            "state": args.state,
            "county": args.county,
            "building_type": args.building_type,
            "area": args.area,
            "num_floors": args.num_floors,
            "height": args.height,
            "n_buildings": args.n_buildings,
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
    logger.info(f"Found {len(buildings)} buildings")
    logger.info(f"Weather file path: {path_to_weather}")
    logger.info("Done!")
