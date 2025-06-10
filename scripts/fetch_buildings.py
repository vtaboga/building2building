import argparse
from datetime import datetime
from building2building.generator.search_idf import search_idf
from building2building.core.run_manager import RunManager
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate IDF files based on specified parameters')
    
    parser.add_argument('--state', '-s', type=str, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, help='County name')
    parser.add_argument('--building-type', '-b', type=str, default="SmallOffice", help='Building type (e.g., SmallOffice)')
    parser.add_argument('--area', '-a', type=float, default=1000, help='Building area in square feet')
    parser.add_argument('--num-floors', '-f', type=int, default=1, help='Number of floors')
    parser.add_argument('--height', type=float, default=None, help='Height of the building')
    parser.add_argument('--n-buildings', '-n', type=int, default=1, help='Number of buildings to search for')
    parser.add_argument('--n-weather-files', '-w', type=int, default=2, help='Number of weather files to download')
    parser.add_argument('--log-level', type=str, default="WARNING", help="the lowest level of logs which will be displayed")
    
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
            "n_weather_files": args.n_weather_files,
        }
    )
    
    # Get logger from run manager
    logger = run_manager.logger

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    logger.info("Searching for building files...")
    buildings, path_to_weather = search_idf(
        state=args.state,
        county=args.county,
        building_type=args.building_type,
        area=args.area,
        num_floors=args.num_floors,
        height=args.height,
        n_buildings=args.n_buildings,
        n_weather_files=args.n_weather_files
    )
    logger.info(f"Found {len(buildings)} buildings")
    logger.info(f"Weather file path: {path_to_weather}")
    logger.info("Done!")
