import io
import logging
import os
import re
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from building2building.env import (
    get_metadata_dir,
    get_unprocessed_idf_dir,
    get_weather_dir,
)

logger = logging.getLogger(__name__)


def download_file_progress(
    url: str, dest: Path, description: str | None = None, verify: bool = False
):
    logger.info(f"Downloading {dest}...")
    response = requests.get(url, stream=True, verify=verify)
    response.raise_for_status()

    # Get total file size from headers
    total_size = int(response.headers.get("content-length", 0))

    block_size = 1 << 16

    if description is None:
        desc = "Downloading"
    else:
        desc = description

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as progress_bar:
        with open(dest, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)


def extract_zip_progress(zip_path: Path, dst_dir: Path, description: str | None = None):
    if description is None:
        desc = "Extracting"
    else:
        desc = description

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.infolist()

        # Calculate total uncompressed size
        total_size = sum(file_info.file_size for file_info in file_list)

        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=desc
        ) as progress_bar:
            for file_info in file_list:
                zip_ref.extract(file_info, dst_dir)
                progress_bar.update(file_info.file_size)


def download_epw(state_code, city_name=None, n_files=2) -> list[Path]:
    """
    Download up to n_files EPW weather files for a specific city and state.
    If city_name is None, downloads any available weather files from the state.

    Args:
        city_name (str or None): Name of the city, or None for any city in the state
        state_code (str): Two-letter state code
        save_dir (str): Directory to save the EPW file
        n_files (int): Number of EPW files to download

    Returns:
        list[str] or str or None: List of paths to the EPW files if successful, None otherwise
    """
    save_dir = get_weather_dir()
    existing_files: list[Path] = list(save_dir.iterdir()) if save_dir.exists() else []

    # Check for already existing files
    found_files = []
    if city_name is None:
        for file in existing_files:
            filename = str(file)
            if file.suffix == ".epw" and (
                f"USA_{state_code.upper()}_" in filename
                or f"_{state_code.upper()}." in filename
            ):
                found_files.append(file)
    else:
        for file in existing_files:
            if (
                file.endswith(".epw")
                and city_name.lower() in file.lower()
                and (
                    f"USA_{state_code.upper()}_" in file
                    or f"_{state_code.upper()}." in file
                )
            ):
                found_files.append(file)

    if len(found_files) >= n_files:
        logger.info(
            f"Found {len(found_files)} existing EPW files for state {state_code}"
        )
        return found_files

    logger.info(
        f"No existing EPW file(s) found for {'state ' + state_code if city_name is None else city_name + ', ' + state_code}. Downloading..."
    )

    base_url = "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/"
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Find up to n_files matching ZIP file links
    zip_file_links: list[str] = []
    for link in soup.find_all("a"):
        href = link.get("href")
        file_name = link.get_text()
        if href and file_name.endswith(".zip"):
            if city_name is None:
                if (
                    f"USA_{state_code.upper()}_" in file_name
                    or f"_{state_code.upper()}." in file_name
                ):
                    zip_file_links.append(base_url + href)
            else:
                if city_name.lower() in file_name.lower() and (
                    f"USA_{state_code.upper()}_" in file_name
                    or f"_{state_code.upper()}." in file_name
                ):
                    zip_file_links.append(base_url + href)
        if len(zip_file_links) >= n_files:
            break

    if not zip_file_links:
        raise Exception(f"No ZIP file found for city: {city_name}, state: {state_code}")

    def download_and_extract_epw(zip_url) -> Path | None:
        response = requests.get(zip_url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            epw_files = [f for f in zip_ref.namelist() if f.endswith(".epw")]
            if epw_files:
                epw_file = epw_files[0]  # Take the first EPW file in the ZIP
                epw_content = zip_ref.read(epw_file)
                epw_path = Path(save_dir, os.path.basename(epw_file))
                with open(epw_path, "wb") as f:
                    f.write(epw_content)
                logger.info(f"Downloaded and extracted EPW file: {epw_path}")
                return epw_path
            else:
                return None

    downloaded_epw_paths: list[Path] = []

    for zip_file_link in zip_file_links:
        if len(downloaded_epw_paths) >= n_files:
            break

        logger.debug(f"Downloading from: {zip_file_link}")
        try:
            maybe_path = download_and_extract_epw(zip_file_link)
            if maybe_path is not None:
                downloaded_epw_paths.append(maybe_path)
        except zipfile.BadZipFile:
            logger.error("Downloaded file is not a valid ZIP file.")
    if len(downloaded_epw_paths) < n_files:
        raise Exception(
            f"Did not find enough files. wanted {n_files} got {len(downloaded_epw_paths)}"
        )
    return downloaded_epw_paths


def get_available_counties() -> list[tuple[str, str]]:
    """
    Scrapes the ESS-DIVE website to get a list of all available county IDF files.

    Returns:
        list[tuple[str, str]]: List of tuples containing (state_code, county_name)
    """
    base_url = "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/Counties_IDF/"

    # Get the webpage content
    response = requests.get(base_url)
    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links that end with _IDF.zip
    zip_links = soup.find_all("a", href=re.compile(r".*_IDF\.zip$"))

    counties: list[tuple[str, str]] = []
    for link in zip_links:
        filename = link["href"]
        # Extract state and county from filename (e.g., "AK_Anchorage_IDF.zip")
        match = re.match(r"([A-Z]{2})_(.+)_IDF\.zip", filename)
        if match:
            state_code = match.group(1)
            county_name = match.group(2)
            counties.append((state_code, county_name))

    logger.debug(f"Found {len(counties)} available counties")
    return sorted(counties)  # Sort by state code and county name


def download_and_extract_county_idf(state_code: str, county_name: str) -> Path:
    """
    Downloads and extracts a county IDF zip file from ESS-DIVE if not already downloaded.

    Args:
        state_code (str): Two-letter state code (e.g., 'AK')
        county_name (str): County name (e.g., 'Anchorage')

    Returns:
        str: Path to the county directory containing IDF files
    """
    # Format the folder name for download
    folder_name = f"{state_code}_{county_name}_IDF"

    # Create output directory if it doesn't exist
    output_dir = get_unprocessed_idf_dir()

    # Create specific folder for this county
    county_dir = output_dir / folder_name

    # Check if directory exists and contains files
    if county_dir.exists() and any(county_dir.iterdir()):
        logger.info(
            f"IDF files for {county_name}, {state_code} already exist in {county_dir}"
        )
        return county_dir

    # If files don't exist, proceed with download
    county_dir.mkdir(parents=True, exist_ok=True)

    # Format the filename for download
    filename = f"{folder_name}.zip"

    # Create the URL
    base_url = "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/Counties_IDF/"
    url = f"{base_url}{filename}"

    # Full path for the zip file
    zip_path = output_dir / filename

    # Download the file
    download_file_progress(
        url, zip_path, description=f"Downloading IDFs for county {county_name}"
    )

    # Extract the zip file to the county-specific folder
    extract_zip_progress(zip_path, county_dir, description="Extracting IDFs")

    # Remove the zip file after extraction
    os.remove(zip_path)
    logger.info(f"Successfully downloaded and extracted {filename}")

    return county_dir


def download_metadata(state: str):
    """
    Downloads the CSV file for the specified state from the metadata URL and stores it in data/metadata directory.
    Creates the directory if it doesn't exist. Skips if already downloaded.

    Args:
        state (str): Two-letter state code (e.g., 'AK')
    """
    metadata_url = "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/MAv1_CSVS/"
    output_dir = get_metadata_dir()
    target_file = f"{state}.csv"

    # Check if file already exists
    file_url = metadata_url + target_file
    file_path = Path(output_dir, target_file)
    if os.path.exists(file_path):
        logger.info(f"Skipping {target_file} - already exists")
        return

    download_file_progress(
        file_url, file_path, description=f"Downloading {target_file}"
    )


def scrap_full_dataset() -> list[Path]:
    """
    Downloads and extracts all available county IDF files from ESS-DIVE.

    Returns:
        list[Path]: List of paths to all downloaded county directories containing IDF files
    """
    logger.info("Starting full dataset download...")

    # Get list of all available counties
    counties = get_available_counties()
    if not counties:
        logger.error("Failed to fetch list of available counties")
        return []

    logger.info(f"Found {len(counties)} counties to download")

    # Store all successful downloads
    downloaded_dirs = []

    # Download each county's data
    for state_code, county_name in counties:
        try:
            logger.info(f"Processing {county_name}, {state_code}...")
            county_dir = download_and_extract_county_idf(state_code, county_name)
            downloaded_dirs.append(county_dir)

        except Exception as e:
            logger.error(f"Failed to download {county_name}, {state_code}: {str(e)}")
            continue

    logger.info(
        f"Download complete. Successfully downloaded {len(downloaded_dirs)} out of {len(counties)} counties"
    )
    return downloaded_dirs
