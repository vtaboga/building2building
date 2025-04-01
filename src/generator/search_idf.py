import pandas as pd
import os
import logging
from src.generator.downloader import download_and_extract_county_idf, download_metadata, download_epw
from src.generator.processing import process_idf, process_metadata

logger = logging.getLogger('generator')

def search_metadata(metadata: pd.DataFrame, building_type: str, area: float, num_floors: int, height: float=None, n_buildings: int=1):
    """
    Search metadata for buildings matching the given criteria.
    """
    # Filter by building type
    filtered = metadata[metadata['BuildingType'] == building_type].copy()
    
    if filtered.empty:
        logger.warning(f"No buildings found of type: {building_type}")
        return []

    # Compute distances for sorting
    filtered['Area_diff'] = (filtered['Area'] - area).abs()
    filtered['Floors_diff'] = (filtered['NumFloors'] - num_floors).abs()
    if height is not None:
        filtered['Height_diff'] = (filtered['Height'] - height).abs()
    else:
        filtered['Height_diff'] = 0 

    # Sort by area difference, then floors, then height
    sorted_filtered = filtered.sort_values(by=['Area_diff', 'Floors_diff', 'Height_diff'])

    # Get the top n_buildings IDs
    result = sorted_filtered['ID'].head(n_buildings).tolist()
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

def search_idf(state: str, county:str, building_type: str, area: float, num_floors: int, height: float, n_buildings: int):
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
        idf_files = search_metadata(metadata, building_type, area, num_floors, height, n_buildings)
        if idf_files:
            logger.info(f"Found {len(idf_files)} matching IDF files: {idf_files}")
        else:
            logger.warning("No matching IDF files found")
            return

        # Process the found IDF files
        processed_files = process_idf(idf_files, state, county)
        logger.info(f"Successfully processed {len(processed_files)} IDF files")

        # Download associated weather files
        download_epw(state_code=state)
        logger.info(f"Successfully downloaded weather files")
        
    except Exception as e:
        logger.error(f"Error during IDF search and processing: {e}")
        raise

    
