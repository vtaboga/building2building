import argparse
from datetime import datetime
from src.generator.search_idf import search_idf
from src.core.logging import setup_logger

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
    
    root_logger.info("Starting IDF generation process...")
    
    # Call search_idf with parsed arguments
    search_idf(
        state=args.state,
        county=args.county,
        building_type=args.building_type,
        area=args.area,
        num_floors=args.num_floors,
        height=args.height,
        n_buildings=args.n_buildings
    )
