import logging
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas
import pandas as pd
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.store import (
    OUTPUT,
    ChildFile,
    Constant,
    Derivation,
    GitClone,
    Realizable,
    Rename,
    derivation,
    realize,
)
from building2building.types import BaseRewardConfig, BuildingConfig
from duckdb import DuckDBPyConnection
from pandas import DataFrame


def housing_archetypes() -> Derivation:
    return GitClone(
        "housing-archetypes",
        "https://github.com/canmet-energy/housing-archetypes",
        "28f443c7697c0f4766eaa2a7b8aebce1ce2026c9",
        bytes.fromhex(
            "0dc405618edd122940d8599c519846d40a711746fbafe8eb0e3f6748b7d8783e"
        ),
    )


idf_path = "data/h2k_files/existing-stock/sd-sa-v11-12-idf-files"
description_path = "data/tables/base_archetype_description.csv"


def housing_archetypes_database() -> Derivation:
    @derivation("index.parquet")
    def index(root: Path):
        output = OUTPUT.get()

        df = duckdb.from_csv_auto(str(root / description_path)).to_df()

        def trans(name: str) -> str:
            p = root / idf_path / name

            return str(p.with_stem(p.stem + "-in").with_suffix(".idf"))

        df = df.assign(filepath=df.filename.apply(trans))

        # There are two different ways québec is written. we defer to the one
        # without the accent because it's easier to work with.
        df.loc[df.region == "QUÉBEC", "region"] = "QUEBEC"

        duckdb.from_df(df).to_parquet(str(output))

    return index(housing_archetypes())


def process(path: Path) -> Derivation:
    return create_complete_pipeline(
        Rename("input.idf", Constant(path)),
        energyplus_path(),
        "24.2.0",
    )


def search_buildings(**query) -> DataFrame:
    ep = energyplus_path()

    db_path = realize(STORE_PATH.get(), housing_archetypes_database())
    db = duckdb.read_parquet(str(db_path)).select(duckdb.StarExpression())

    for k, v in query.items():
        db = db.filter(duckdb.ColumnExpression(k) == duckdb.ConstantExpression(v))

    df = db.to_df()

    def trans(path: str):
        return lambda: create_complete_pipeline(
            Constant(Path(path)),
            ep,
            src_version="24.2.0",
        )

    return df.assign(derivation_thunk=df["filepath"].apply(trans))


# def search_config() -> BuildingConfig:
#     b = realize(STORE_PATH.get(), search_buildings().iloc[0].derivation_thunk())
#     return BuildingConfig(
#         path_to_building=b,
#         path_to_weather=w,
#         reward_config=BaseRewardConfig(1000.0),
#         energy_weight=1.0,
#         eplus_output_dir=Path(tempfile.mkdtemp()),
#         warmup_phases=3,
#     )


logging.basicConfig(level=logging.DEBUG)


def test():
    realize(
        STORE_PATH.get(), search_buildings(region="QUEBEC").iloc[0].derivation_thunk()
    )
