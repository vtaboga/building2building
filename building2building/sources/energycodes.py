import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.store import (
    OUTPUT,
    Constant,
    Derivation,
    DownloadFile,
    ExtractZip,
    Rename,
    derivation,
    realize,
)
from building2building.types import BaseRewardConfig, BuildingConfig
from pandas import DataFrame


def ASHRAE901_all() -> Derivation:
    return ExtractZip(
        DownloadFile(
            "ASHRAE901_all.zip",
            "https://www.energycodes.gov/sites/default/files/2023-10/ASHRAE901_all.zip",
            bytes.fromhex(
                "de35252dada89f6e24f6007e24c2c1796a047c294707f491b65e81cc7ee212ab"
            ),
        ),
    )


@derivation("idf_index.parquet")
def index_buildings(input: Path):
    dst = OUTPUT.get()
    directory = input
    records = []

    pattern = re.compile(r"^ASHRAE901_([^_]+)_STD(\d{4})_([^.]+)\.idf$")

    # Get all .idf files in the directory
    directory_path = Path(directory)
    for file_path in directory_path.glob("*.idf"):
        filename = file_path.name
        match = pattern.match(filename)

        if match:
            building_type = match.group(1)
            year = match.group(2)
            place = match.group(3)

            records.append((building_type, int(year), place, str(file_path)))

    # Sort by building type, then year, then place for consistent ordering
    records.sort(key=lambda x: (x[0], x[1], x[2]))

    df = DataFrame(records, columns=["building_type", "year", "place", "path"])
    df.to_parquet(str(dst))


@derivation("epw_index.parquet")
def index_weathers(input: Path):
    dst = OUTPUT.get()
    directory = input
    records = []

    pattern = re.compile(r"^USA_([^.]+)_([^.]+).*$")

    # Get all .idf files in the directory
    for file_path in directory.glob("*.epw"):
        filename = file_path.name
        match = pattern.match(filename)

        if match:
            state = match.group(1)
            county = match.group(2)

            records.append((state, county, str(file_path)))

    # Sort by building type, then year, then place for consistent ordering
    records.sort(key=lambda x: (x[0], x[1], x[2]))

    df = DataFrame(records, columns=["state", "county", "path"])
    df.to_parquet(str(dst))


def search_buildings(
    building_type: str | None = None, year: int | None = None, place: str | None = None
) -> DataFrame:
    idf_index = realize(STORE_PATH.get(), index_buildings(ASHRAE901_all()))
    db = duckdb.from_parquet(str(idf_index))

    expr = db
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

    def trans(path):
        return lambda: create_complete_pipeline(
            Constant(path),
            ep,
            src_version="22.1.0",
        )

    return df.assign(derivation_thunk=df["path"].apply(trans))


def search_weathers(
    state: str | None = None,
    county: str | None = None,
) -> DataFrame:
    epw_index = realize(STORE_PATH.get(), index_weathers(ASHRAE901_all()))
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

    def trans(path: str):
        return lambda: Constant(Path(path))

    return df.assign(derivation_thunk=df["path"].apply(trans))


def search_config(
    building_type: str | None = None,
    year: int | None = None,
    place: str | None = None,
    state: str | None = None,
    county: str | None = None,
) -> BuildingConfig:
    b = realize(
        STORE_PATH.get(),
        search_buildings(
            building_type=building_type,
            year=year,
            place=place,
        )
        .iloc[0]
        .derivation_thunk(),
    )
    w = realize(
        STORE_PATH.get(),
        search_weathers(
            state=state,
            county=county,
        )
        .iloc[0]
        .derivation_thunk(),
    )

    return BuildingConfig(
        path_to_building=b,
        path_to_weather=w,
        reward_config=BaseRewardConfig(1000.0),
        energy_weight=1.0,
        eplus_output_dir=Path(tempfile.mkdtemp()),
        warmup_phases=3,
    )
