from dataclasses import dataclass
from pathlib import Path

import b2b.sources.geo as geo
import duckdb
import geopandas as gpd
import pandas as pd
from b2b.env import STORE_PATH
from b2b.store import (
    OUTPUT,
    Derivation,
    DownloadFile,
    ExtractZip,
    derivation,
)
from shapely.geometry import Point


def weather_files() -> Derivation:
    url = "https://data.nrel.gov/system/files/156/Buildstock_TMY3_FIPS-1678817889.zip"

    d = DownloadFile(
        "weather_files.zip",
        url,
        bytes.fromhex(
            "b9ae1223d75f3f0c42a0056ac391b7a2aaec9a4cd69312f0ed89ff23759bbfc7"
        ),
    )
    e = ExtractZip(d)
    return e


@derivation("weathers.parquet")
def WeatherTable(root: Path):
    dst = OUTPUT.get()

    df = duckdb.from_csv_auto(str(root / "map_FIPs_TMY3s.csv")).to_df()

    l = list(map(lambda x: str(root) + f"/{x}.epw", df.original_FIP.tolist()))

    df["filepath"] = l

    df.to_parquet(str(dst))


def weather_table() -> Derivation:
    return WeatherTable(weather_files())
