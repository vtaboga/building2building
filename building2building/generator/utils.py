import asyncio
import logging
import zipfile
from pathlib import Path
from typing import List, Tuple

import aiohttp
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from building2building.generator.downloader import (
    download_file_progress,
    extract_zip_progress,
)

logger = logging.getLogger(__name__)
