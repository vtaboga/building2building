import requests
import logging
import asyncio
import aiohttp
from typing import List, Tuple

logger = logging.getLogger('generator')

def get_county_from_coords(lat, lon):
    """
    Returns the US county name for given latitude and longitude using FCC API.
    Formats to fit dataset format: removes the last word and replaces spaces/hyphens with underscores.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        str: Formatted county name if found, else None
    """
    url = "https://geo.fcc.gov/api/census/block/find"
    params = {
        "latitude": lat,
        "longitude": lon,
        "format": "json"
    }

    try:
        logger.debug(f"Querying FCC API for coordinates: ({lat}, {lon})")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        county_name = data.get('County', {}).get('name')
        
        if county_name:
            # Remove the last word (usually 'County') and split remaining words
            words = county_name.split()[:-1]
            if words:
                # Join remaining words and replace spaces/hyphens with underscore
                formatted_name = '_'.join(''.join(words).replace('-', '_').split())
                logger.debug(f"Found county: {county_name} -> formatted as: {formatted_name}")
                return formatted_name
            
        logger.warning(f"No county name found for coordinates ({lat}, {lon})")
        return None
        
    except requests.RequestException as e:
        logger.error(f"Request error for coordinates ({lat}, {lon}): {e}")
    except Exception as e:
        logger.error(f"Unexpected error while getting county for coordinates ({lat}, {lon}): {e}")
    
    return None

async def get_county_async(session: aiohttp.ClientSession, lat: float, lon: float) -> str:
    """
    Asynchronously get county name for a single coordinate pair.
    """
    try:
        async with session.get(
            "https://geo.fcc.gov/api/census/block/find",
            params={
                "latitude": lat,
                "longitude": lon,
                "format": "json"
            }
        ) as response:
            if response.status == 200:
                data = await response.json()
                county_name = data.get('County', {}).get('name')
                if county_name:
                    # Remove the last word (usually 'County') and split remaining words
                    words = county_name.split()[:-1]
                    if words:
                        # Join remaining words and replace spaces/hyphens with underscore
                        return '_'.join(''.join(words).replace('-', '_').split())
            return None
    except Exception as e:
        logger.error(f"Error getting county for coordinates ({lat}, {lon}): {e}")
        return None

async def get_counties_batch_async(coords_list: List[Tuple[float, float]], max_concurrent: int = 50) -> List[str]:
    """
    Asynchronously get county names for all coordinate pairs using a connection pool.
    
    Args:
        coords_list: List of (latitude, longitude) tuples
        max_concurrent: Maximum number of concurrent connections
    
    Returns:
        List of county names corresponding to the coordinates
    """
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max_concurrent),
        timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes total timeout
    ) as session:
        tasks = []
        for lat, lon in coords_list:
            tasks.append(get_county_async(session, lat, lon))
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        counties = []
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            logger.debug(f"Processing chunk {i//chunk_size + 1}/{(len(tasks) + chunk_size - 1)//chunk_size}")
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            counties.extend([
                result if not isinstance(result, Exception) else None 
                for result in chunk_results
            ])
            
        return counties

def get_counties_from_coords_batch(coords_list: List[Tuple[float, float]]) -> List[str]:
    """
    Get county names for all coordinate pairs.
    
    Args:
        coords_list: List of (latitude, longitude) tuples
    
    Returns:
        List of county names corresponding to the coordinates
    """
    try:
        # Create new event loop for async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(get_counties_batch_async(coords_list))
    finally:
        loop.close()
