import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import requests
import logging
import asyncio
import aiohttp
from typing import List, Tuple
import os
import zipfile

logger = logging.getLogger('generator')

def download_county_boundaries():
    """
    Download US county boundaries from Census Bureau
    """

    zip_file_name = os.path.join("data", "metadata", "counties.zip")

    # Option 1: Cartographic boundaries (smaller, simplified)
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"

    # Option 2: Full TIGER/Line boundaries (more detailed, larger)
    # url = "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"


    if not os.path.exists(zip_file_name):
        logger.info("Downloading county boundaries...")
        response = requests.get(url, verify=False)
        with open(zip_file_name, 'wb') as f:
            f.write(response.content)

    boundaries_dir = os.path.join("data", "metadata", "counties")

    if not os.path.exists(boundaries_dir):
        logger.info("Extracting county boundaries...")

        # Extract the shapefile
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(boundaries_dir)

    return boundaries_dir

def load_county_boundaries(data_dir):
    """
    Load county boundaries into a GeoDataFrame
    """
    # Find the .shp file in the extracted directory
    shp_files = [f for f in os.listdir(data_dir) if f.endswith('.shp')]
    if not shp_files:
        raise FileNotFoundError("No shapefile found in the extracted data")

    shp_path = os.path.join(data_dir, shp_files[0])

    logger.info(f"Loading {shp_path}...")
    counties = gpd.read_file(shp_path)

    # Ensure we're using WGS84 (EPSG:4326) for lat/lon coordinates
    if counties.crs != 'EPSG:4326':
        counties = counties.to_crs('EPSG:4326')

    logger.info(f"Loaded {len(counties)} counties")
    logger.info(f"Columns available: {list(counties.columns)}")
    return counties

def fast_county_lookup(lat_lon_pairs, counties_gdf):
    coords_df = pd.DataFrame(lat_lon_pairs, columns=["latitude", "longitude"])
    geometry = [Point(lon, lat) for lat, lon in lat_lon_pairs]
    points_gdf = gpd.GeoDataFrame(coords_df, geometry=geometry, crs="EPSG:4326")
    logger.info("computing expensive sjoin...")
    result = gpd.sjoin(points_gdf, counties_gdf, how="left", predicate="within")
    return result.drop(["geometry"], axis=1)

def batch_county_lookup(lat_lon_pairs, counties_gdf):
    """
    Perform batch county lookup for thousands of coordinate pairs

    Parameters:
    - lat_lon_pairs: List of tuples [(lat1, lon1), (lat2, lon2), ...]
    - counties_gdf: GeoDataFrame with county boundaries

    Returns:
    - List of county information for each coordinate pair
    """
    results = []

    logger.info(f"Processing {len(lat_lon_pairs)} coordinate pairs...")

    for i, (lat, lon) in enumerate(lat_lon_pairs):
        if i % 1000 == 0:
            logger.info(f"Processed {i/(len(lat_lon_pairs)-1)} of all coordinates")

        # Create point geometry
        point = Point(lon, lat)  # Note: Point(lon, lat) not (lat, lon)

        # Find which county contains this point
        containing_counties = counties_gdf[counties_gdf.geometry.contains(point)]

        if len(containing_counties) > 0:
            county_info = containing_counties.iloc[0]
            results.append({
                'latitude': lat,
                'longitude': lon,
                'county_fips': county_info.get('GEOID', ''),
                'county_name': county_info.get('NAME', ''),
                'state_fips': county_info.get('STATEFP', ''),
                'state_name': county_info.get('STATE_NAME', '')  # May not be available in all datasets
            })
        else:
            # Point not found in any county (e.g., water, outside US)
            results.append({
                'latitude': lat,
                'longitude': lon,
                'county_fips': None,
                'county_name': None,
                'state_fips': None,
                'state_name': None
            })

    print("Batch lookup complete!")
    return results


def get_counties_from_coords_batch(coords_list: List[Tuple[float, float]]) -> List[str]:
    data_dir = download_county_boundaries()
    counties = load_county_boundaries(data_dir)
    counties.sindex # create an index to accelerate inclusion calculations
    results = fast_county_lookup(coords_list, counties)
    return list(results["NAME"])
