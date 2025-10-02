import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from building2building.env import STORE_PATH
from building2building.store import Derivation, DownloadFile, ExtractZip, realize
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def county_boundaries() -> Derivation:
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"

    return ExtractZip(
        DownloadFile(
            "county_borders.zip",
            url,
            bytes.fromhex(
                "99d6597b1fc7767deef62e01d28d8b5dcbd578e151855f7dc0d173cbf5bf0868"
            ),
        ),
    )


def load_county_boundaries(data_dir: Path):
    """
    Load county boundaries into a GeoDataFrame
    """
    # Find the .shp file in the extracted directory
    shp_files = [f for f in data_dir.iterdir() if f.suffix == ".shp"]
    if not shp_files:
        raise FileNotFoundError("No shapefile found in the extracted data")

    shp_path = shp_files[0]

    logger.debug(f"Loading {shp_path}...")
    counties = gpd.read_file(shp_path)

    # Ensure we're using WGS84 (EPSG:4326) for lat/lon coordinates
    if counties.crs != "EPSG:4326":
        counties = counties.to_crs("EPSG:4326")

    logger.debug(f"Loaded {len(counties)} counties")
    logger.debug(f"Columns available: {list(counties.columns)}")
    return counties


def fast_county_lookup(lat_lon_pairs, counties_gdf):
    coords_df = pd.DataFrame(lat_lon_pairs, columns=["latitude", "longitude"])
    geometry = [Point(lon, lat) for lat, lon in lat_lon_pairs]
    points_gdf = gpd.GeoDataFrame(coords_df, geometry=geometry, crs="EPSG:4326")
    logger.info("computing expensive sjoin...")
    result = gpd.sjoin(points_gdf, counties_gdf, how="left", predicate="within")
    return result.drop(["geometry"], axis=1)


def get_counties_from_coords_batch(coords_list: list[tuple[float, float]]) -> list[str]:
    """From a list of (latitude, longitude) pairs, compute the list of"""
    data_dir = build(STORE_PATH.get(), county_boundaries())
    counties = load_county_boundaries(data_dir)
    counties.sindex  # create an index to accelerate inclusion calculations
    results = fast_county_lookup(coords_list, counties)
    return list(results["NAME"])


def find_closest(
    lat1: list[float], lon1: list[float], lat2: list[float], lon2: list[float]
) -> list[int]:
    """Return, for each point on the left, the index of the closest point on the
    right."""

    df1 = pd.DataFrame({"latitude": lat1, "longitude": lon1})
    gdf1 = gpd.GeoDataFrame(
        df1, geometry=gpd.points_from_xy(df1.latitude, df1.longitude)
    )

    df2 = pd.DataFrame({"latitude": lat2, "longitude": lon2})
    gdf2 = gpd.GeoDataFrame(
        df2, geometry=gpd.points_from_xy(df2.latitude, df2.longitude)
    )

    gdf1 = gdf1.set_crs("EPSG:4326")
    gdf2 = gdf2.set_crs("EPSG:4326")
    # gdf1 = gdf1.to_crs("EPSG:3857")
    # gdf2 = gdf2.to_crs("EPSG:3857")

    result = gpd.sjoin_nearest(gdf1, gdf2, how="left")
    # Sometimes, there are multiple weather files at the exact same location.
    # This forces one to be chosen.
    result = result.loc[~result.index.duplicated(keep="first")]

    return list(result.index_right)
