"""Source module for the multizones_reference_buildings dataset.

This dataset contains 6000 parametrically varied EnergyPlus epJSON files
(1000 per building type) generated from ASHRAE 90.1-2022 prototypes with
Latin Hypercube Sampling over envelope, fenestration, infiltration, and
geometry parameters.

Building types: Warehouse, HotelSmall, RetailStandalone, RestaurantFastFood,
OfficeMedium, OfficeSmall.

Layout inside the zip::

    dataset/
        metadata_0.csv  .. metadata_5.csv
        weather/
            *.epw
        1.epJSON .. 6000.epJSON
"""

from __future__ import annotations

import csv
import io
import logging
import traceback
import json
import zipfile
from pathlib import Path
from typing import Any, Literal

import duckdb
from pandas import DataFrame

from b2b.env import STORE_PATH
from b2b.pipeline import (
    add_hvac_meters,
    add_outdoor_air_meters,
    extract_discovery_metadata,
    make_controllable,
    modify_run_period,
    modify_timestep,
)
from b2b.store import (
    OUTPUT,
    Constant,
    Derivation,
    DownloadFile,
    ExtractFromZip,
    Realizable,
    Rename,
    derivation,
    realize,
)
from b2b.types import (
    BaseRewardConfig,
    BuildingConfig,
)

logger = logging.getLogger(__name__)

BuildingType = Literal[
    "Warehouse",
    "HotelSmall",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]


def dataset_zip() -> Derivation:
    return DownloadFile(
        "multizones_reference_buildings.zip",
        "https://huggingface.co/datasets/vtaboga/multizones_reference_buildings/resolve/main/multizones_reference_buildings.zip",
        bytes.fromhex(
            "85e437d1fbbd095edd5ba3d4206fa8036e5043ba937318b9f3725d97e9d1eb4e"
        ),
    )


@derivation("multizones_index.parquet")
def table_index(root_zip: Path) -> None:
    out = OUTPUT.get()

    all_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(root_zip) as zf:
        for name in sorted(zf.namelist()):
            if not name.startswith("dataset/metadata_") or not name.endswith(".csv"):
                continue
            with zf.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    row["building_id"] = int(row["building_id"])
                    row["epjson_filename"] = f"dataset/{row['building_id']}.epJSON"
                    all_rows.append(row)

    df = DataFrame(all_rows)
    df.to_parquet(str(out))


def _build_control_derivation(
    root_zip: Realizable,
    epjson_filename: str,
) -> Any:
    """Build a control-ready epJSON from a raw dataset epJSON.

    The dataset already ships epJSON v25.1.0 so no upgrade/conversion is
    needed — we only add meters, set the timestep, and make controllable.
    """
    current: Derivation = ExtractFromZip(root_zip, epjson_filename)
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)
    current = modify_timestep(current, timesteps_per_hour=4)
    current = modify_run_period(
        current,
        begin_day_of_month=1, begin_month=1,
        end_day_of_month=31, end_month=12,
    )
    current = Rename("building.epjson", current)
    return make_controllable(current)


def search_buildings(
    building_type: BuildingType | None = None,
    place: str | None = None,
    building_id: int | None = None,
    **query: Any,
) -> DataFrame:
    root_zip = dataset_zip()
    index = realize(STORE_PATH.get(), table_index(root_zip))

    db = duckdb.from_parquet(str(index))

    if building_id is not None:
        db = db.filter(
            duckdb.ColumnExpression("building_id")
            == duckdb.ConstantExpression(building_id)
        )
    if building_type is not None:
        db = db.filter(
            duckdb.ColumnExpression("building_type")
            == duckdb.ConstantExpression(building_type)
        )
    if place is not None:
        db = db.filter(
            duckdb.FunctionExpression("lower", duckdb.ColumnExpression("place"))
            == duckdb.ConstantExpression(place.lower())
        )
    for k, v in query.items():
        if isinstance(v, str):
            db = db.filter(
                duckdb.FunctionExpression("lower", duckdb.ColumnExpression(k))
                == duckdb.ConstantExpression(v.lower())
            )
        elif isinstance(v, (int, float)):
            db = db.order(f"abs({k} - {v})")

    df = db.to_df()

    def trans(epjson_filename: str):
        return lambda: _build_control_derivation(root_zip, epjson_filename)

    return df.assign(derivation_thunk=df["epjson_filename"].apply(trans))


def search_configs(
    config: dict | object | None = None,
    n: int = 2,
    eplus_output_dir: Path = Path("eplus_out"),
) -> list[BuildingConfig]:
    """Return BuildingConfigs for selected buildings."""
    cfg: dict[str, Any] = {}
    if config is not None:
        if isinstance(config, dict):
            cfg = config
        else:
            try:
                from omegaconf import OmegaConf  # type: ignore

                result = OmegaConf.to_container(config, resolve=True)
                cfg = result if isinstance(result, dict) else {}
            except Exception:
                cfg = {}

    bldg_query = cfg.get("bldg", {})
    if isinstance(bldg_query, dict) and "bldg" in bldg_query and isinstance(bldg_query["bldg"], dict):
        bldg_query = bldg_query["bldg"]

    rows = search_buildings(**bldg_query)
    root_zip = dataset_zip()

    configs: list[BuildingConfig] = []
    for _, row in rows.iterrows():
        if len(configs) >= n:
            break

        source_meta: dict[str, object] = {
            "source": "multizones_reference_buildings",
            "building_id": int(row.building_id),
            "building_type": str(row.building_type),
            "place": str(row.place),
        }

        try:
            epw_derivation = ExtractFromZip(root_zip, f"dataset/{row.weather_file}")

            control_derivation = row.derivation_thunk()
            epjson, hvac_equipment = realize(STORE_PATH.get(), control_derivation)

            metadata = realize(
                STORE_PATH.get(),
                extract_discovery_metadata(Constant(epjson), epw_derivation),
            )

            epw = realize(STORE_PATH.get(), epw_derivation)

            reward_section = cfg.get("reward", {}) if isinstance(cfg, dict) else {}
            if not isinstance(reward_section, dict):
                reward_section = {}
            energy_weight = reward_section.get("energy_weight", 0.0)

            configs.append(
                BuildingConfig(
                    path_to_building=epjson,
                    path_to_weather=epw,
                    reward_config=BaseRewardConfig(energy_weight=energy_weight),
                    hvac_equipment=hvac_equipment,
                    eplus_output_dir=eplus_output_dir,
                    warmup_phases=metadata.warmup_phases,
                    area=metadata.net_conditioned_area,
                    source_metadata=source_meta,
                )
            )
        except Exception as e:
            eplus_output_dir.mkdir(parents=True, exist_ok=True)
            err_path = eplus_output_dir / "pipeline_errors.jsonl"
            record = {
                "source_metadata": source_meta,
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            try:
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                logger.exception("Failed to write pipeline error record to %s", err_path)

            logger.warning(
                "Failed to build BuildingConfig for building_id=%s: %s",
                source_meta.get("building_id", "unknown"),
                e,
            )
            continue

    return configs
