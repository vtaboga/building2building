import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
from b2b.env import STORE_PATH, energyplus_path
from b2b.pipeline import create_complete_pipeline
from b2b.store import (
    OUTPUT,
    Constant,
    Derivation,
    DownloadFile,
    ExtractFromZip,
    ExtractZip,
    Rename,
    derivation,
    realize,
)
from b2b.types import BaseRewardConfig, BuildingConfig
from pandas import DataFrame


def ASHRAE901_all_zip() -> Derivation:
    return DownloadFile(
        "ASHRAE901_all.zip",
        "https://www.energycodes.gov/sites/default/files/2023-10/ASHRAE901_all.zip",
        bytes.fromhex(
            "de35252dada89f6e24f6007e24c2c1796a047c294707f491b65e81cc7ee212ab"
        ),
    )


@derivation("idf_index.parquet")
def index_buildings(input_zip: Path):
    dst = OUTPUT.get()
    records = []

    idf_pattern = re.compile(r".*\.idf$")

    with zipfile.ZipFile(input_zip) as zip_ref:
        info_list = zip_ref.infolist()

    pattern = re.compile(r"^ASHRAE901_([^_]+)_STD(\d{4})_([^.]+)\.idf$")

    for info in info_list:
        if not idf_pattern.match(info.filename):
            continue

        match = pattern.match(info.filename)
        if match:
            building_type = match.group(1)
            year = match.group(2)
            place = match.group(3)

            records.append((building_type, int(year), place, str(info.filename)))

    # Sort by building type, then year, then place for consistent ordering
    records.sort(key=lambda x: (x[0], x[1], x[2]))
    df = DataFrame(records, columns=["building_type", "year", "place", "filename"])
    df = df.reset_index(drop=False)
    df.to_parquet(str(dst))


@derivation("epw_index.parquet")
def index_weathers(input_zip: Path):
    dst = OUTPUT.get()
    records = []

    pattern = re.compile(r"^USA_([^.]+)_([^.]+).*\.epw$")

    with zipfile.ZipFile(input_zip) as zip_ref:
        info_list = zip_ref.infolist()

    # Get all .epw files in the directory
    for file_info in info_list:
        match = pattern.match(file_info.filename)

        if match:
            state = match.group(1)
            county = match.group(2)

            records.append((state, county, str(file_info.filename)))

    # Sort by building type, then year, then place for consistent ordering
    records.sort(key=lambda x: (x[0], x[1], x[2]))

    df = DataFrame(records, columns=["state", "county", "filename"])
    df.to_parquet(str(dst))


# Copy-pasted from set(search_buildings().building_type)
BuildingType = Literal[
    "RetailStripmall",
    "HotelLarge",
    "ApartmentMidRise",
    "Warehouse",
    "ApartmentHighRise",
    "HotelSmall",
    "OfficeLarge",
    "SchoolPrimary",
    "RetailStandalone",
    "SchoolSecondary",
    "RestaurantFastFood",
    "OfficeMedium",
    "Hospital",
    "OutPatientHealthCare",
    "OfficeSmall",
    "RestaurantSitDown",
]


def search_buildings(
    index: int | None = None,
    building_type: BuildingType | None = None,
    year: int | None = None,
    place: str | None = None,
) -> DataFrame:
    zip_derivation = ASHRAE901_all_zip()

    idf_index = realize(STORE_PATH.get(), index_buildings(zip_derivation))
    db = duckdb.from_parquet(str(idf_index))

    expr = db
    if index is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("index") == duckdb.ConstantExpression(index)
        )

    if building_type is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("building_type")
            == duckdb.ConstantExpression(building_type)
        )

    if year is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("year") == duckdb.ConstantExpression(year)
        )

    if place is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("place") == duckdb.ConstantExpression(place)
        )

    df = expr.to_df()

    ep = energyplus_path()

    def trans(name: str):
        return lambda: create_complete_pipeline(
            ExtractFromZip(zip_derivation, name),
            ep,
            src_version="22.1.0",
        )

    return df.assign(derivation_thunk=df["filename"].apply(trans))


def search_weathers(
    state: str | None = None,
    county: str | None = None,
) -> DataFrame:
    zip_derivation = ASHRAE901_all_zip()
    epw_index = realize(STORE_PATH.get(), index_weathers(zip_derivation))
    db = duckdb.from_parquet(str(epw_index))
    expr = db
    if state is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("state") == duckdb.ConstantExpression(state)
        )

    if county is not None:
        expr = expr.filter(
            duckdb.ColumnExpression("county") == duckdb.ConstantExpression(county)
        )

    df = expr.to_df()

    def trans(filename: str):
        return lambda: ExtractFromZip(zip_derivation, filename)

    return df.assign(derivation_thunk=df["filename"].apply(trans))


def search_config(
    building_type: BuildingType | None = None,
    year: int | None = None,
    place: str | None = None,
    state: str | None = None,
    county: str | None = None,
) -> BuildingConfig:
    buildings = search_buildings(
        building_type=building_type,
        year=year,
        place=place,
    )

    b, equipment = realize(
        STORE_PATH.get(),
        buildings.iloc[0].derivation_thunk(),
    )
    weathers = search_weathers(
        state=state,
        county=county,
    )

    w = realize(
        STORE_PATH.get(),
        weathers.iloc[0].derivation_thunk(),
    )

    return BuildingConfig(
        path_to_building=b,
        path_to_weather=w,
        reward_config=BaseRewardConfig(1.0),
        eplus_output_dir=Path(tempfile.mkdtemp()),
        warmup_phases=3,
        area=1000.0,
        hvac_equipment=equipment,
        source_metadata={
            "building": buildings.iloc[0],
            "weather": weathers.iloc[0],
        },
    )
