from dataclasses import dataclass
from pathlib import Path

import building2building.sources.geo as geo
import duckdb
import geopandas as gpd
import pandas as pd
from building2building.env import STORE_PATH
from building2building.store import (
    BaseDerivation,
    Child,
    Derivation,
    DownloadFile,
    ExtractZip,
    build,
    default_hash,
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
    e = ExtractZip("weather_files", d)
    return e


@dataclass
class WeatherTable(BaseDerivation):
    root: Derivation

    def name(self) -> str:
        return "weathers.parquet"

    def build(self, dst: Path, deps):
        df = duckdb.from_csv_auto(str(deps["root"] / "map_FIPs_TMY3s.csv")).to_df()

        l = list(
            map(lambda x: str(deps["root"]) + f"/{x}.epw", df.original_FIP.tolist())
        )

        df["filepath"] = l

        df.to_parquet(str(dst))


def weather_table() -> Derivation:
    return WeatherTable(weather_files())


# p = build(STORE_PATH.get(), weather_table())

# db = duckdb.from_csv_auto(str(p))
# lat_lon_pairs = db.select(
#     duckdb.ColumnExpression("station_lat"), duckdb.ColumnExpression("station_lon")
# ).fetchall()


# lat, lon = zip(*lat_lon_pairs)

# counties = geo.get_counties_from_coords_batch(lat_lon_pairs)

# r = geo.find_closest(lat, lon, lat, lon)
