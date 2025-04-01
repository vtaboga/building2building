import requests
from bs4 import BeautifulSoup
import os
import re
import zipfile
import io
from pathlib import Path
import logging

logger = logging.getLogger('generator')

def download_epw(state_code, city_name=None, save_dir="data/weather"):
    """
    Download EPW weather file for a specific city and state.
    If city_name is None, downloads any available weather file from the state.
    
    Args:
        city_name (str or None): Name of the city, or None for any city in the state
        state_code (str): Two-letter state code
        save_dir (str): Directory to save the EPW file
    
    Returns:
        str or None: Path to the EPW file if successful, None otherwise
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Check existing files in the save directory
    existing_files = os.listdir(save_dir) if os.path.exists(save_dir) else []
    
    # Special handling for None city_name - look for any file from the state
    if city_name is None:
        for file in existing_files:
            if file.endswith('.epw') and (f"USA_{state_code.upper()}_" in file or f"_{state_code.upper()}." in file):
                logger.info(f"Found existing EPW file for state {state_code}: {os.path.join(save_dir, file)}")
                return os.path.join(save_dir, file)
    else:
        for file in existing_files:
            if file.endswith('.epw') and city_name.lower() in file.lower() and (
                f"USA_{state_code.upper()}_" in file or f"_{state_code.upper()}." in file):
                logger.info(f"EPW file already exists: {os.path.join(save_dir, file)}")
                return os.path.join(save_dir, file)
    
    logger.info(f"No existing EPW file found for {'state '+state_code if city_name is None else city_name+', '+state_code}. Downloading...")
    
    base_url = "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/"
    
    # Get the webpage content
    response = requests.get(base_url)
    if response.status_code != 200:
        logger.error("Failed to access the webpage.")
        return None
    
    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the ZIP file that matches the criteria
    zip_file_link = None
    
    # Look for links containing the required information
    for link in soup.find_all("a"):
        href = link.get("href")
        file_name = link.get_text()
        
        if href and file_name.endswith(".zip"):
            if city_name is None:
                # For None city_name, match any file from the state with exact state code pattern
                if f"USA_{state_code.upper()}_" in file_name or f"_{state_code.upper()}." in file_name:
                    zip_file_link = base_url + href
                    logger.debug(f"Found matching ZIP file for state {state_code}: {file_name}")
                    break
            else:
                # Original behavior for specific city with exact state code pattern
                if city_name.lower() in file_name.lower() and (
                    f"USA_{state_code.upper()}_" in file_name or f"_{state_code.upper()}." in file_name):
                    zip_file_link = base_url + href
                    logger.debug(f"Found matching ZIP file: {file_name}")
                    break

    if zip_file_link:
        # Download the ZIP file
        logger.debug(f"Downloading from: {zip_file_link}")
        response = requests.get(zip_file_link)
        
        if response.status_code == 200:
            # Extract EPW file from the ZIP
            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    # Find the EPW file in the ZIP
                    epw_files = [f for f in zip_ref.namelist() if f.endswith('.epw')]
                    
                    if epw_files:
                        epw_file = epw_files[0]  # Take the first EPW file
                        
                        # Extract the EPW file
                        epw_content = zip_ref.read(epw_file)
                        
                        # Save the EPW file
                        epw_path = os.path.join(save_dir, os.path.basename(epw_file))
                        with open(epw_path, "wb") as f:
                            f.write(epw_content)
                        
                        logger.info(f"Downloaded and extracted EPW file: {epw_path}")
                        return epw_path
                    else:
                        logger.warning("No EPW file found in the ZIP archive.")
            except zipfile.BadZipFile:
                logger.error("Downloaded file is not a valid ZIP file.")
        else:
            logger.error(f"Failed to download ZIP file. Status code: {response.status_code}")
    else:
        logger.warning(f"No ZIP file found for city: {city_name}, state: {state_code}")
    
    return None


def get_available_counties() -> list[tuple[str, str]]:
    """
    Scrapes the ESS-DIVE website to get a list of all available county IDF files.
    
    Returns:
        list[tuple[str, str]]: List of tuples containing (state_code, county_name)
    """
    base_url = "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/Counties_IDF/"
    
    try:
        # Get the webpage content
        response = requests.get(base_url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links that end with _IDF.zip
        zip_links = soup.find_all('a', href=re.compile(r'.*_IDF\.zip$'))
        
        counties = []
        for link in zip_links:
            filename = link['href']
            # Extract state and county from filename (e.g., "AK_Anchorage_IDF.zip")
            match = re.match(r'([A-Z]{2})_(.+)_IDF\.zip', filename)
            if match:
                state_code = match.group(1)
                county_name = match.group(2)
                counties.append((state_code, county_name))
        
        logger.debug(f"Found {len(counties)} available counties")
        return sorted(counties)  # Sort by state code and county name
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error accessing website: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return []

def download_and_extract_county_idf(state_code: str, county_name: str) -> str:
    """
    Downloads and extracts a county IDF zip file from ESS-DIVE if not already downloaded.
    
    Args:
        state_code (str): Two-letter state code (e.g., 'AK')
        county_name (str): County name (e.g., 'Anchorage')
    
    Returns:
        str: Path to the county directory containing IDF files
    """
    # Format the folder name
    folder_name = f"{state_code}_{county_name}_IDF"
    
    # Create output directory if it doesn't exist
    output_dir = Path("data/idf")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create specific folder for this county
    county_dir = output_dir / folder_name
    
    # Check if directory exists and contains files
    if county_dir.exists() and any(county_dir.iterdir()):
        logger.info(f"IDF files for {county_name}, {state_code} already exist in {county_dir}")
        return county_dir
    
    # If files don't exist, proceed with download
    county_dir.mkdir(exist_ok=True)
    
    # Format the filename
    filename = f"{folder_name}.zip"
    
    # Create the URL
    base_url = "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/Counties_IDF/"
    url = f"{base_url}{filename}"
    
    # Full path for the zip file
    zip_path = output_dir / filename
    
    try:
        # Download the file
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file to the county-specific folder
        logger.debug(f"Extracting {filename} to {folder_name}/...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(county_dir)
        
        # Remove the zip file after extraction
        os.remove(zip_path)
        logger.info(f"Successfully downloaded and extracted {filename}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
    except zipfile.BadZipFile:
        logger.error(f"Error: {filename} is not a valid zip file")
        if zip_path.exists():
            os.remove(zip_path)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if zip_path.exists():
            os.remove(zip_path)

    return county_dir


def download_metadata(state: str):
    """
    Downloads the CSV file for the specified state from the metadata URL and stores it in data/metadata directory.
    Creates the directory if it doesn't exist. Skips if already downloaded.

    Args:
        state (str): Two-letter state code (e.g., 'AK')
    """
    metadata_url = "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/MAv1_CSVS/"
    output_dir = "data/metadata"
    target_file = f"{state}.csv"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if file already exists
    file_path = os.path.join(output_dir, target_file)
    if os.path.exists(file_path):
        logger.info(f"Skipping {target_file} - already exists")
        return

    try:
        # Download the file
        file_url = metadata_url + target_file
        response = requests.get(file_url)
        response.raise_for_status()

        # Save the file
        with open(file_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded: {target_file}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading metadata for {state}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

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
    
    logger.info(f"Download complete. Successfully downloaded {len(downloaded_dirs)} out of {len(counties)} counties")
    return downloaded_dirs





