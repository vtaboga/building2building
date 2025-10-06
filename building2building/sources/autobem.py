import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import building2building.sources.geo as geo
import building2building.sources.nrel as nrel
import duckdb
import pandas as pd
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.sources.geo import get_counties_from_coords_batch
from building2building.store import (
    OUTPUT,
    ChildFile,
    Constant,
    Derivation,
    DownloadFile,
    ExtractZip,
    derivation,
    realize,
)
from building2building.types import BaseRewardConfig, BuildingConfig

logger = logging.getLogger(__name__)

StateCode = Literal[
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]


def validate_state_code(code: str) -> StateCode:
    if code in get_args(StateCode):
        return code  # type: ignore
    else:
        raise Exception(f"{code} is not a valid state code")


def manifest() -> Derivation:
    return DownloadFile(
        "manifest-md5.txt",
        "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/manifest-md5.txt",
        bytes.fromhex(
            "e9dde3952b2a8cf0a52833704616eb15d1cceed05e146477c1c3f60f8b84fad1"
        ),
    )


def state_metadata_hashes() -> dict[StateCode, bytes]:
    filename = realize(STORE_PATH.get(), manifest())

    md5_hashes = {}
    with open(filename, "r") as f:
        for line in f:
            if ".csv" in line:
                md5, path = line.strip().split("  ")
                state = path.split("/")[-1].replace(".csv", "")
                md5_hashes[state] = bytes.fromhex(md5)

    return md5_hashes


def state_counties_hashes() -> dict[StateCode, dict[str, bytes]]:
    filename = realize(STORE_PATH.get(), manifest())

    md5_hashes: dict[StateCode, dict[str, bytes]] = {}
    with open(filename, "r") as f:
        for line in f:
            if "IDF.zip" in line:
                md5, path = line.strip().split("  ")
                # Extract state_county from path like ./data/Counties_IDF/TX_Crosby_IDF.zip
                state_county = path.split("/")[-1].replace("_IDF.zip", "")
                state, county = state_county.split("_", 1)
                # In the database, there are no spaces, but in the filename,
                # there are.
                county = county.replace("_", " ")

                state_code = validate_state_code(state)
                md5_hashes.setdefault(state_code, dict())[county] = bytes.fromhex(md5)
    return md5_hashes


def underscore(str) -> str:
    return str.replace(" ", "_")


def state_counties_zips() -> dict[StateCode, dict[str, Derivation]]:
    base_url = "https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/Counties_IDF/"
    hashes = state_counties_hashes()
    return {
        state: {
            county: DownloadFile(
                f"{state}_{underscore(county)}_IDF.zip",
                base_url + f"{state}_{underscore(county)}_IDF.zip",
                hash,
                hasher_factory=hashlib.md5,
            )
            for county, hash in county_hash.items()
        }
        for state, county_hash in hashes.items()
    }


def state_counties_extracted() -> dict[StateCode, dict[str, Derivation]]:
    zips = state_counties_zips()
    return {
        state: {county: ExtractZip(step) for county, step in county_hash.items()}
        for state, county_hash in zips.items()
    }


def metadata_csv(state: StateCode) -> Derivation:
    metadata_hashes = state_metadata_hashes()
    hash = metadata_hashes[state]

    metadata_url = f"https://tier2.ess-dive.lbl.gov/doi-10-15485-2283980/data/MAv1_csvs/{state}.csv"

    return DownloadFile(f"{state}.csv", metadata_url, hash, hashlib.md5)


@derivation("metadata-for-county.parquet")
def MetadataKeepCounty(
    metadata: Path,
    idf_files: Path,
    county: str,
):
    dst = OUTPUT.get()

    def exists(id: int) -> bool:
        building_path = idf_files / f"{id}.idf"

        return building_path.exists()

    ids = [
        row.ID
        for row in duckdb.from_parquet(str(metadata))
        .filter(duckdb.ColumnExpression("County") == duckdb.ConstantExpression(county))
        .select(duckdb.ColumnExpression("Id"))
        .to_df()
        .itertuples()
    ]
    existing_ids = list(filter(exists, ids))

    # Read the parquet file as a relation first
    relation = duckdb.read_parquet(str(metadata))
    # Then use SQL on that relation
    ids_str = ",".join(map(str, existing_ids))
    result = relation.filter(f"Id IN ({ids_str})")
    result.to_parquet(str(dst))


@derivation("metadata.parquet")
def MetadataComputeCounty(
    metadata: Path,
):
    dst = OUTPUT.get()
    # Read the CSV file
    df = pd.read_csv(metadata)

    # Split the Centroid column into latitude and longitude
    df[["Latitude", "Longitude"]] = df["Centroid"].str.split("/", expand=True)

    # Convert to float
    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)

    coords_list = list(zip(df["Latitude"], df["Longitude"]))
    counties = get_counties_from_coords_batch(coords_list)

    # Add counties to dataframe
    df["County"] = counties

    # Save the updated CSV file
    df.to_parquet(dst)


def search_buildings(
    state: StateCode,
    county: str,
    building_id: int | None = None,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
) -> pd.DataFrame:
    county_extracted_map = state_counties_extracted()[state]

    if county not in county_extracted_map:
        raise Exception(
            f"invalid county. valid choices: {list(county_extracted_map.keys())}"
        )

    extracted = county_extracted_map[county]

    metadata_path = realize(
        STORE_PATH.get(),
        MetadataKeepCounty(
            MetadataComputeCounty(metadata_csv(state)), extracted, county
        ),
    )

    # We query the database

    conn = duckdb.from_parquet(str(metadata_path))
    expr = conn.select(duckdb.StarExpression())

    if building_id is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("ID") == duckdb.ConstantExpression(building_id)
        )

    if county is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("County") == duckdb.ConstantExpression(county)
        )

    if building_type is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("BuildingType")
            == duckdb.ConstantExpression(building_type)
        )

    if num_floors is not None:
        expr = expr.order(f"ABS(NumFloors - {num_floors})")

    if area is not None:
        expr = expr.order(f"ABS(Area - {area})")

    if height is not None:
        expr = expr.order(f"ABS(Height - {height})")

    df = expr.to_df()

    ep = energyplus_path()

    # We define the pipeline for all the files

    def trans(building_id):
        return lambda: create_complete_pipeline(
            ChildFile(extracted, f"{building_id}.idf"),
            ep,
            src_version="9.4.0",
        )

    return df.assign(derivation_thunk=df["ID"].apply(trans))


def search_config(
    state: StateCode,
    county: str,
    building_id: int | None = None,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
    eplus_output_dir: Path = Path("eplus_out"),
) -> BuildingConfig:
    buildings = search_buildings(
        state,
        county,
        building_id=building_id,
        building_type=building_type,
        area=area,
        num_floors=num_floors,
        height=height,
    )
    matching_buildings = buildings.iloc[[0]]

    weather_table = realize(STORE_PATH.get(), nrel.weather_table())

    weather_df = duckdb.from_parquet(str(weather_table)).to_df()

    weather_lat: list = weather_df.station_lat.tolist()
    weather_lon: list = weather_df.station_lon.tolist()

    closest = geo.find_closest(
        matching_buildings.Latitude.tolist(),
        matching_buildings.Longitude.tolist(),
        weather_lat,
        weather_lon,
    )

    (building_id,) = matching_buildings.ID.tolist()
    (weather_file,) = weather_df.iloc[closest].filepath.tolist()

    b = matching_buildings.iloc[0]

    building_path = realize(
        STORE_PATH.get(), matching_buildings.iloc[0].derivation_thunk()
    )

    return BuildingConfig(
        building_path,
        Path(weather_file),
        BaseRewardConfig(b.Area, 1.0),
        eplus_output_dir=eplus_output_dir,
        # Empirically, this works for this dataset.
        warmup_phases=5,
    )
