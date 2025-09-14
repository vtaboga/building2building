import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.store import (
    BaseDerivation,
    Derivation,
    DownloadFile,
    ExtractZip,
    Symlink,
    build,
)


def ASHRAE901_all() -> Derivation:
    return ExtractZip(
        "all_buildings",
        DownloadFile(
            "ASHRAE901_all.zip",
            "https://www.energycodes.gov/sites/default/files/2023-10/ASHRAE901_all.zip",
            bytes.fromhex(
                "de35252dada89f6e24f6007e24c2c1796a047c294707f491b65e81cc7ee212ab"
            ),
        ),
    )


@dataclass
class IndexBuildings(BaseDerivation):
    input: Derivation

    def name(self):
        return "idf_index.parquet"

    def build(self, dst: Path, deps: dict[str, Path]):
        directory = deps["input"]
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

        df = pd.DataFrame(records, columns=["building_type", "year", "place", "path"])
        df.to_parquet(str(dst))


@dataclass
class IndexWeathers(BaseDerivation):
    input: Derivation

    def name(self):
        return "epw_index.parquet"

    def build(self, dst: Path, deps: dict[str, Path]):
        directory = deps["input"]
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

        df = pd.DataFrame(records, columns=["state", "county", "path"])
        df.to_parquet(str(dst))


def search_buildings(
    building_type: str | None = None, year: int | None = None, place: str | None = None
) -> pd.DataFrame:
    idf_index = build(STORE_PATH.get(), IndexBuildings(ASHRAE901_all()))
    db = duckdb.from_parquet(str(idf_index))

    expr = db
    if building_type != None:
        expr = expr.filter(
            duckdb.ColumnExpression("building_type")
            == duckdb.ConstantExpression(building_type)
        )

    if year != None:
        expr = expr.filter(
            duckdb.ColumnExpression("year") == duckdb.ConstantExpression(year)
        )

    if place != None:
        expr = expr.filter(
            duckdb.ColumnExpression("place") == duckdb.ConstantExpression(place)
        )

    df = expr.to_df()

    def trans(path):
        return create_complete_pipeline(
            Symlink(Path(path)),
            energyplus_path(),
            transitions=[
                "22.1.0-to-22.2.0",
                "22.2.0-to-23.1.0",
                "23.1.0-to-23.2.0",
                "23.2.0-to-24.1.0",
            ],
        )

    return df.assign(derivation=df["path"].apply(trans))


def search_weathers(
    state: str | None = None,
    county: str | None = None,
):
    epw_index = build(STORE_PATH.get(), IndexWeathers(ASHRAE901_all()))
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

    return df.assign(derivation=df["path"].apply(lambda path: Symlink(Path(path))))


def search_config() -> BuildingConfig:
    b = build(STORE_PATH.get(), search_buildings().iloc[0].derivation)
    w = build(STORE_PATH.get(), search_weathers().iloc[0].derivation)

    return BuildingConfig(
        path_to_building=b,
        path_to_weather=w,
        reward_config=BaseRewardConfig(1000.0),
        energy_weight=1.0,
        eplus_output_dir=Path(tempfile.mkdtemp()),
        warmup_phases=3,
    )
