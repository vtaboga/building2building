import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Tuple, TypeAlias, reveal_type

import duckdb
import pandas as pd
from minergym.ontology import Ontology

import building2building.generator.processing as processing
from building2building.generator.downloader import (
    download_and_extract_county_idf,
    download_epw,
    download_metadata,
)
from building2building.types import BuildingCharacteristics

logger = logging.getLogger(__name__)


@dataclass
class BuildingMetadata:
    building_type: str
    num_floors: int
    area: float
    height: float


def search_metadata(
    metadata_path: Path,
    county: str | None = None,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
    n_buildings: int = 1,
) -> list[tuple[int, BuildingMetadata]]:
    """
    Search metadata for buildings matching the given criteria in order:
    1. Building type (required)
    2. Number of floors (if provided)
    3. Area (if provided)
    4. Height (if provided)
    """

    where_conditions: list[tuple[str, str]] = []

    if county is not None:
        where_conditions.append(("County", county))

    if building_type is not None:
        where_conditions.append(("BuildingType", building_type))

    order_by_parts: list[tuple[str, float]] = []

    if num_floors is not None:
        order_by_parts.append(("NumFloors", num_floors))

    if area is not None:
        order_by_parts.append(("Area", area))

    if height is not None:
        order_by_parts.append(("Height", height))

    if metadata_path.suffix == ".csv":
        read_function = "read_csv"
    else:
        read_function = "read_parquet"

    query_parts = [f"SELECT * FROM {read_function}(?)"]

    params: list[Any] = [str(metadata_path)]

    if where_conditions:
        where_clauses = []
        for col, val in where_conditions:
            where_clauses.append(f"{col} = ?")
            params.append(val)
        query_parts.append(f"WHERE {' AND '.join(where_clauses)}")

    if order_by_parts:
        distance_terms = []
        for col, val in order_by_parts:
            distance_terms.append(f"ABS({col} - ?) ASC")
            params.append(val)
        distance_expr = " , ".join(distance_terms)
        query_parts.append(f"ORDER BY {distance_expr}")

    query_parts.append("LIMIT ?")
    params.append(n_buildings)

    query = " ".join(query_parts)

    res = duckdb.query(query, params=params)
    out = []
    for row in res.to_df().itertuples():
        out.append(
            (
                int(row.ID),
                BuildingMetadata(
                    building_type=(row.BuildingType),
                    num_floors=(row.NumFloors),
                    area=(row.Area),
                    height=(row.Height),
                ),
            )
        )

    return out


EPJSONProcessor: TypeAlias = Callable[[Path, Path], Any]


def hash_processors(processors: list[EPJSONProcessor], hash_length=16) -> str:
    processor_names = [p.__qualname__ for p in processors]
    combined = "".join(processor_names)
    hash_object = hashlib.sha256(combined.encode("utf-8"))
    hash_hex = hash_object.hexdigest()[:hash_length]
    return hash_hex


def process_idf(
    in_path: Path,
    tmp_path: Path,
    out_path: Path,
    processors: list[EPJSONProcessor] = [],
):
    all_processors = (
        [processing.specialized_upgrade_idf(t) for t in processing.transitions]
        + [processing.convert_idf]
        + processors
        + [processing.add_setpoint_control_to_epjson]
    )

    def path_until(processors):
        h = hash_processors(processors)
        return (tmp_path / in_path.with_suffix("").name).with_suffix("." + h)

    tmp_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(in_path, path_until([]))

    for i, proc in enumerate(all_processors):
        logger.debug(f"executing processor {proc}")

        artifact_in_path = path_until(all_processors[:i])
        artifact_out_path = path_until(all_processors[: i + 1])
        if artifact_out_path.exists():
            logger.info(f"Skipping {in_path} - already processed")
            continue

        proc(artifact_in_path, artifact_out_path)

    final_artifact_path = path_until(all_processors)
    shutil.copy(final_artifact_path, out_path)


def search_idf(
    state: str,
    county: str,
    n_buildings: int,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
    processors: list[EPJSONProcessor] = [],
) -> list[Tuple[Path, BuildingCharacteristics]]:
    """
    Search and process IDF files matching the specified criteria.

    Args:
        state (str): Two-letter state code
        county (str): County name
        building_type (str): Type of building to search for
        area (float): Target floor area
        num_floors (int): Target number of floors
        height (float|None): Target building height
        n_buildings (int): Number of buildings to return
        n_weather_files (int): Number of weather files to return
        keep_original (bool): If True, keep original IDF files after processing

    Returns:
        Tuple[list[Tuple[str, dict]], list[str]]: List of (building_path, characteristics) tuples and list of weather file paths
    """
    logger.info(f"Starting IDF search for {building_type} in {county}, {state}")
    logger.debug(
        f"Search criteria: area={area}, floors={num_floors}, height={height}, n={n_buildings}"
    )

    # Download idf files if not already downloaded
    county_dir = download_and_extract_county_idf(state, county)
    logger.debug(f"Using county directory: {county_dir}")

    building_files: list[Path] = []
    building_infos: list[BuildingCharacteristics] = []
    # Load metadata
    download_metadata(state=state)

    metadata_path = processing.process_metadata(state=state)

    # Search for matching IDF files
    matching_buildings = search_metadata(
        metadata_path,
        county,
        n_buildings=n_buildings,
        building_type=building_type,
        area=area,
        num_floors=num_floors,
        height=height,
    )
    if matching_buildings:
        logger.info(
            f"Found {len(matching_buildings)} matching IDF files: {matching_buildings}"
        )
    else:
        raise Exception("No matching IDF files found")

    for building_id, building_info in matching_buildings:
        building_path = Path(
            "data", "idf", f"{state}_{county}_IDF", f"{building_id}.idf"
        )

        if len(processors) != 0:
            processors_hash = "." + hash_processors(processors)
        else:
            processors_hash = ""

        tmp_dir = Path("data", "processed_buildings", state, county)
        processed_path = (
            tmp_dir / building_path.with_suffix(f"{processors_hash}.epJSON").name
        )

        if not processed_path.exists():
            processed_path.parent.mkdir(exist_ok=True, parents=True)
            process_idf(building_path, tmp_dir, processed_path, processors)

        info_path = processed_path.with_suffix(".json")
        if not info_path.exists():
            ont = Ontology.from_json(processed_path)
            # Add zone lists to characteristics
            zone_list: list[str] = [n.toPython() for n in ont.zones()]
            building_characteristics = BuildingCharacteristics(
                building_info.building_type,
                building_info.num_floors,
                building_info.area,
                building_info.height,
                zone_list,
            )

            building_characteristics.save_json(info_path)
            logger.info(f"Saved characteristics to {info_path}")
        else:
            building_characteristics = BuildingCharacteristics.load_json(info_path)

        building_files.append(processed_path)
        building_infos.append(building_characteristics)

    return list(zip(building_files, building_infos))
