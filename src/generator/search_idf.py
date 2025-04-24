import pandas as pd
import os
import logging
from src.generator.downloader import download_and_extract_county_idf, download_metadata, download_epw
from src.generator.processing import process_idf, process_metadata

logger = logging.getLogger('generator')

def search_metadata(metadata: pd.DataFrame, building_type: str, area: float, num_floors: int, height: float=None, n_buildings: int=1):
    """
    Search metadata for buildings matching the given criteria in order:
    1. Building type (required)
    2. Number of floors (if provided)
    3. Area (if provided)
    4. Height (if provided)
    """
    # Filter by building type (required)
    filtered = metadata[metadata['BuildingType'] == building_type].copy()
    
    if filtered.empty:
        logger.warning(f"No buildings found of type: {building_type}")
        return []

    # Initialize difference columns with 0 (no difference)
    filtered['Floors_diff'] = 0
    filtered['Area_diff'] = 0
    filtered['Height_diff'] = 0

    # Calculate differences for provided criteria
    if num_floors is not None:
        filtered['Floors_diff'] = (filtered['NumFloors'] - num_floors).abs()
    
    if area is not None:
        filtered['Area_diff'] = (filtered['Area'] - area).abs()
    
    if height is not None:
        filtered['Height_diff'] = (filtered['Height'] - height).abs()

    # Sort by criteria in specified order: floors, area, height
    sorted_filtered = filtered.sort_values(
        by=['Floors_diff', 'Area_diff', 'Height_diff']
    )

    # Get the top n_buildings IDs and their characteristics
    result = [
        (row['ID'], {
            'building_type': row['BuildingType'],
            'num_floors': row['NumFloors'],
            'area': row['Area'],
            'height': row['Height']
        })
        for _, row in sorted_filtered.head(n_buildings).iterrows()
    ]
    logger.debug(f"Found {len(result)} matching buildings")
    return result

def load_metadata(state: str, county: str=None):
    """
    Load and filter metadata for the specified state and county.
    """
    try:
        metadata_path = os.path.join("data", "metadata", f"{state}.csv")
        metadata = pd.read_csv(metadata_path)
        logger.debug(f"Loaded metadata for state {state}")
        
        if county is not None:
            metadata = metadata[metadata['County'] == county]
            logger.debug(f"Filtered metadata for county {county}: {len(metadata)} entries")
        return metadata
        
    except FileNotFoundError:
        logger.error(f"Metadata file not found for state: {state}")
        raise
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        raise

def search_idf(state: str, county: str, building_type: str, area: float, num_floors: int, height: float, n_buildings: int):
    """
    Search and process IDF files matching the specified criteria.
    """
    logger.info(f"Starting IDF search for {building_type} in {county}, {state}")
    logger.debug(f"Search criteria: area={area}, floors={num_floors}, height={height}, n={n_buildings}")

    # Download idf files if not already downloaded
    county_dir = download_and_extract_county_idf(state, county)
    logger.debug(f"Using county directory: {county_dir}")

    # Load metadata
    try:
        download_metadata(state=state)
        process_metadata(state=state)
        metadata = load_metadata(state, county=county)
        logger.debug(f"Loaded metadata with shape: {metadata.shape}")

        # Search for matching IDF files
        building_files = search_metadata(metadata, building_type, area, num_floors, height, n_buildings)
        if building_files:
            logger.info(f"Found {len(building_files)} matching IDF files: {building_files}")
        else:
            logger.warning("No matching IDF files found")
            return

        # Process the found IDF files
        processed_files = process_idf(building_files, state, county)
        logger.info(f"Successfully processed {len(processed_files)} IDF files")

        # Download associated weather files
        weather_file = download_epw(state_code=state)
        logger.info(f"Successfully downloaded weather files")
        
    except Exception as e:
        logger.error(f"Error during IDF search and processing: {e}")
        raise

    # Create tuples of (path_to_building, dict_of_characteristics)

    base_path = os.path.join("data/processed_buildings", f"{state}", f"{county}")
    
    
    building_files = [
        (os.path.join(base_path, f"{building[0]}.epJSON"), building[1])
        for building in building_files
    ]

    return building_files, weather_file

    
