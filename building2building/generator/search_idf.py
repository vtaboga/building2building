import pandas as pd
import duckdb
import os
import logging
from building2building.generator.downloader import download_and_extract_county_idf, download_metadata, download_epw
import building2building.generator.processing as processing
from typing import Tuple, TypeAlias, Callable
from pathlib import Path
from minergym.ontology import Ontology
import json
import hashlib
import tempfile
import shutil
from functools import partial

logger = logging.getLogger(__name__)


def search_metadata(metadata: pd.DataFrame, building_type: str, area: float|None, num_floors: int|None, height: float|None=None, n_buildings: int=1) -> list[tuple[int, dict]]:
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

def load_metadata(metadata_path: Path, county: str|None=None) -> pd.DataFrame:
    """
    Load and filter metadata for the specified state and county.
    """
    if county is None:
        query = f"SELECT * FROM '{metadata_path}'"
    else:
        logger.debug(f"Will filtered metadata for county {county}")
        query = f"SELECT * FROM '{metadata_path}' WHERE County = '{county}'"


    logger.debug(f"Executing query `{query}`")
    df = duckdb.query(query).to_df()
    logger.debug(f"Loaded metadata.")

    return df

EPJSONProcessor: TypeAlias = Callable[[Path], Path]

def hash_processors(processors: list[EPJSONProcessor], hash_length=16) -> str:
    processor_names = [p.__qualname__ for p in processors]
    combined = "".join(processor_names)
    hash_object = hashlib.sha256(combined.encode('utf-8'))
    hash_hex = hash_object.hexdigest()[:hash_length]
    return hash_hex

def process_idf(in_path: Path, tmp_path:Path, out_path: Path, processors: list[EPJSONProcessor]=[]):

    all_processors = (
        [processing.specialized_upgrade_idf(t) for t in processing.transitions]
        + [processing.convert_idf]
        + processors
        + [processing.add_setpoint_control_to_epjson]
    )

    def path_until(processors):
        h = hash_processors(processors)
        return (tmp_path / in_path.with_suffix("").name).with_suffix("."+h)

    tmp_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(in_path, path_until([]))

    for i, proc in enumerate(all_processors):
        logger.debug(f"executing processor {proc}")

        artifact_in_path = path_until(all_processors[:i])
        artifact_out_path = path_until(all_processors[:i+1])
        if artifact_out_path.exists():
            logger.info(f"Skipping {in_path} - already processed")
            continue

        proc(artifact_in_path, artifact_out_path)

    final_artifact_path = path_until(all_processors)
    shutil.copy(final_artifact_path, out_path)

def search_idf(state: str, county: str, building_type: str, area: float, num_floors: int, height: float|None, n_buildings: int, n_weather_files: int, processors:list[EPJSONProcessor]=[]) -> Tuple[list[Tuple[Path, dict]], list[Path]]:
    """
    Search and process IDF files matching the specified criteria.
    """
    logger.info(f"Starting IDF search for {building_type} in {county}, {state}")
    logger.debug(f"Search criteria: area={area}, floors={num_floors}, height={height}, n={n_buildings}")

    # Download idf files if not already downloaded
    county_dir = download_and_extract_county_idf(state, county)
    logger.debug(f"Using county directory: {county_dir}")

    building_files: list[Path] = []
    building_infos: list[dict] = []
    # Load metadata
    try:
        download_metadata(state=state)

        metadata_path = processing.process_metadata(state=state)
        metadata = load_metadata(metadata_path, county=county)
        logger.debug(f"Loaded metadata with shape: {metadata.shape}")

        # Search for matching IDF files
        matching_buildings = search_metadata(metadata, building_type, area, num_floors, height, n_buildings)
        if matching_buildings:
            logger.info(f"Found {len(matching_buildings)} matching IDF files: {matching_buildings}")
        else:
            raise Exception("No matching IDF files found")


        for building_id, building_info in matching_buildings:
            building_path = Path("data", "idf", f"{state}_{county}_IDF", f"{building_id}.idf")

            if len(processors) != 0:
                processors_hash = "." + hash_processors(processors)
            else:
                processors_hash = ""

            tmp_dir = Path("data", "processed_buildings", state, county)
            processed_path = tmp_dir / building_path.with_suffix(f"{processors_hash}.epJSON").name

            if not processed_path.exists():
                process_idf(building_path, tmp_dir, processed_path, processors)

            info_path = processed_path.with_suffix(".json")
            if not info_path.exists():
                ont = Ontology.from_json(processed_path)
                # Add zone lists to characteristics
                zone_list = [n.toPython() for n in ont.zones()]
                building_info["zone_lists"] = zone_list

                with open(info_path, 'w') as json_file:
                    json.dump(building_info, json_file, indent=4)
                    logger.info(f"Saved characteristics to {info_path}")
            else:
                with open(info_path, "r") as f:
                    building_info = json.load(f)

            building_files.append(processed_path)
            building_infos.append(building_info)

        # Download associated weather files
        weather_files = download_epw(state_code=state, n_files=n_weather_files)

    except Exception as e:
        logger.error(f"Error during IDF search and processing: {e}")
        raise

    # Create tuples of (path_to_building, dict_of_characteristics)

    return list(zip(building_files, building_infos)), weather_files
