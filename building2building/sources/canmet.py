from pathlib import Path

import building2building.sources.oneclimate as oneclimate
import duckdb
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline
from building2building.store import (
    OUTPUT,
    Constant,
    Derivation,
    GitClone,
    Rename,
    derivation,
    realize,
)
from building2building.types import BaseRewardConfig, BuildingConfig
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
        if v is None:
            continue
        db = db.filter(duckdb.ColumnExpression(k) == duckdb.ConstantExpression(v))

    df = db.to_df()

    def trans(path: str):
        return lambda: create_complete_pipeline(
            Constant(Path(path)),
            ep,
            src_version="24.2.0",
        )

    return df.assign(derivation_thunk=df["filepath"].apply(trans))


def search_config(
    province: str | None = None,
    city: str | None = None,
    eplus_output_dir: Path = Path("eplus_out"),
) -> BuildingConfig:
    buildings = search_buildings(province=province, location=city)

    matching_buildings = buildings.iloc[[0]]

    # province_code: oneclimate.ProvinceCode | None

    if province is not None:
        province_code = oneclimate.ProvinceCode = oneclimate.province_to_code[province]
    else:
        province_code = None

    weather_df = oneclimate.search_weathers(province=province_code, city=city)

    # Select best matching EPW: prefer CWEC2020, then TMYx, then TMY
    def score(fname: str) -> tuple[int, int, int]:
        f = fname.lower()
        return (
            1 if "cwec2020" in f else 0,
            1 if "tmyx" in f else 0,
            1 if "tmy" in f else 0,
        )

    best_row = max(weather_df.iterrows(), key=lambda r: score(r[1].url))
    weather_path = realize(STORE_PATH.get(), best_row[1].derivation_thunk())
    building_path = realize(
        STORE_PATH.get(), matching_buildings.iloc[0].derivation_thunk()
    )

    return BuildingConfig(
        path_to_building=building_path,
        path_to_weather=weather_path,
        reward_config=BaseRewardConfig(1000, 1.0),  # TODO: handle floor area in reward
        eplus_output_dir=eplus_output_dir,
        # Empirically, this works for this dataset.
        warmup_phases=1,  # TODO: handle warmup
    )
