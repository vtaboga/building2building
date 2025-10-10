from importlib.resources import files
from pathlib import Path

import duckdb
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.store import (
    OUTPUT,
    Constant,
    Derivation,
    ExtractZip,
    LocalFile,
    derivation,
    realize,
)
from building2building.types import BaseRewardConfig, BuildingConfig
from pandas import DataFrame


def extracted() -> Derivation:
    place = files("building2building.sources.data") / "hydroquebec.zip"
    if not isinstance(place, Path):
        raise Exception("error")

    return ExtractZip(LocalFile(place))


def table_index():
    @derivation("table.parquet")
    def build_database(root: Path):
        out = OUTPUT.get()

        df = duckdb.from_csv_auto(
            str(root / "2025-10-09_building-stock-100-mila.csv")
        ).to_df()

        df = df.assign(
            epw_path=df.weather_station_epw_filepath.apply(
                lambda name: str(root / "weather" / name)
            )
        )
        df = df.assign(
            idf_path=[
                str(root / "IDFs" / f"building_{i}.idf") for i in range(1, len(df) + 1)
            ]
        )

        df = df.drop(columns=["geometry_roof_pitch"])

        df.to_parquet(str(out))

    return build_database(extracted())


def search_weathers() -> DataFrame:
    index = realize(STORE_PATH.get(), table_index())
    return duckdb.from_parquet(str(index)).to_df()


def search_buildings() -> DataFrame:
    index = realize(STORE_PATH.get(), table_index())

    db = duckdb.from_parquet(str(index))

    df = db.to_df()

    ep = energyplus_path()

    def trans(idf_path):
        return lambda: create_complete_pipeline(
            Constant(Path(idf_path)),
            ep,
            src_version="24.2.0",
        )

    df = df.assign(derivation_thunk=df.idf_path.apply(trans))

    return df


def search_config(eplus_output_dir=Path("eplus_out")) -> BuildingConfig:
    buildings = search_buildings()

    row = buildings.iloc[0]

    epw = Path(row.epw_path)
    epjson = realize(STORE_PATH.get(), row.derivation_thunk())
    return BuildingConfig(
        epjson, epw, BaseRewardConfig(1000.0, 1.0), eplus_output_dir, warmup_phases=10
    )
