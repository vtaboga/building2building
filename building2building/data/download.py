"""HuggingFace dataset download and cache management.

Downloads building data from the ``vtaboga/building2building_dataset``
HuggingFace repository.  Each building type is stored as a separate zip
archive, enabling partial downloads.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

REPO_ID = "vtaboga/building2building_dataset"
REVISION = "main"

BuildingType = Literal[
    "SingleFamilyHouse",
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

ALL_BUILDING_TYPES: list[BuildingType] = [
    "SingleFamilyHouse",
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]


def _cache_dir() -> Path:
    """Return the local directory where extracted buildings are stored."""
    return Path.home() / ".cache" / "building2building"


def download_metadata() -> Path:
    """Download and return path to the unified ``metadata.parquet``."""
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename="metadata.parquet",
            repo_type="dataset",
            revision=REVISION,
        )
    )


def download_splits() -> Path:
    """Download and return path to ``splits.json``."""
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename="splits.json",
            repo_type="dataset",
            revision=REVISION,
        )
    )


def download_building_type(building_type: BuildingType) -> Path:
    """Download and extract buildings for a given building type.

    Uses ``huggingface_hub`` caching: repeated calls are no-ops if already
    downloaded.

    Returns:
        Path to the extracted directory containing individual building
        folders.
    """
    cache = _cache_dir()
    extract_dir = cache / building_type
    marker = extract_dir / ".download_complete"

    if marker.exists():
        return extract_dir

    zip_name = f"{building_type}.zip"
    logger.info("Downloading %s from %s ...", zip_name, REPO_ID)
    zip_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=zip_name,
            repo_type="dataset",
            revision=REVISION,
        )
    )

    logger.info("Extracting %s to %s ...", zip_name, extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    marker.touch()
    return extract_dir


def get_building_path(building_type: BuildingType, building_id: str) -> Path:
    """Return path to a specific building's directory.

    Downloads the building type archive if not already cached.

    Args:
        building_type: The building type (e.g. ``"OfficeSmall"``).
        building_id: The building ID (e.g. ``"OfficeSmall-0042"``).

    Returns:
        Path to the directory containing ``building.epjson``,
        ``equipment.json``, and ``metadata.json``.
    """
    type_dir = download_building_type(building_type)
    building_dir = type_dir / building_id
    if not building_dir.is_dir():
        raise FileNotFoundError(
            f"Building {building_id!r} not found in {type_dir}. "
            f"Available: {sorted(p.name for p in type_dir.iterdir() if p.is_dir())[:5]}..."
        )
    return building_dir
