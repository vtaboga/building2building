"""Source module for the multizones_reference_buildings dataset.

This dataset contains 6000 parametrically varied EnergyPlus epJSON files
(1000 per building type) generated from ASHRAE 90.1-2022 prototypes with
Latin Hypercube Sampling over envelope, fenestration, infiltration, and
geometry parameters.

Building types: Warehouse, RetailStandalone, RestaurantFastFood,
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
import random
import zipfile
from pathlib import Path
from typing import Any, Literal, Sequence

import duckdb
from pandas import DataFrame

from building2building.env import STORE_PATH
from building2building.pipeline import (
    add_hvac_meters,
    add_outdoor_air_meters,
    extract_discovery_metadata,
    make_controllable,
    modify_run_period,
    modify_timestep,
)
from building2building.store import (
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
from building2building.types import BuildingConfig, TaskConfig, reward_config_from_dict

logger = logging.getLogger(__name__)

BuildingType = Literal[
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

SPLIT_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_split_ids(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    *,
    split_data_dir: Path | None = None,
) -> list[int]:
    base_dir = split_data_dir if split_data_dir is not None else SPLIT_DATA_DIR
    path = base_dir / f"{building_type}_{split}_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    ids: list[int] = json.loads(path.read_text(encoding="utf-8"))
    return ids


def building_id_from_split_index(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    split_index: int,
) -> int:
    ids = load_split_ids(building_type, split)
    if split_index < 0 or split_index >= len(ids):
        raise IndexError(
            f"Index {split_index} out of range for {building_type}/{split} "
            f"(has {len(ids)} buildings, valid: 0..{len(ids) - 1})"
        )
    return int(ids[split_index])


def building_ids_from_split_indices(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    split_indices: Sequence[int],
) -> list[int]:
    return [
        building_id_from_split_index(building_type, split, int(idx))
        for idx in split_indices
    ]


def sample_building_ids(
    building_type: BuildingType,
    split: Literal["train", "test", "test_small"],
    n: int,
    *,
    seed: int | None = None,
    replace: bool = False,
) -> list[int]:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be int >= 0, got {n!r}")
    ids = load_split_ids(building_type, split)
    rng = random.Random(seed)
    if n == 0:
        return []
    if replace:
        return [int(rng.choice(ids)) for _ in range(n)]
    if n > len(ids):
        raise ValueError(
            f"Cannot sample n={n} without replacement from only {len(ids)} ids"
        )
    return [int(x) for x in rng.sample(ids, k=n)]


def dataset_zip() -> Derivation:
    return DownloadFile(
        "multizones_reference_buildings.zip",
        "https://huggingface.co/datasets/vtaboga/multizones_reference_buildings/resolve/main/multizones_reference_buildings.zip",
        bytes.fromhex(
            "66b94393c129d78a8271e70da805ca48a8af9fdecb1204d2d9fb95398d3786de"
        ),
    )


@derivation("multizones_index.parquet")
def table_index(root_zip: Path) -> None:
    out = OUTPUT.get()

    all_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(root_zip) as zf:
        for name in sorted(zf.namelist()):
            basename = name.rsplit("/", 1)[-1]
            if not basename.startswith("metadata") or not basename.endswith(".csv"):
                continue
            with zf.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    row["building_id"] = int(row["building_id"])
                    row["epjson_filename"] = f"{row['building_id']}.epJSON"
                    all_rows.append(row)

    df = DataFrame(all_rows)
    df.to_parquet(str(out))


def _build_control_derivation(
    root_zip: Realizable,
    epjson_filename: str,
    run_period_name: str,
    timesteps_per_hour: int = 12,
) -> Any:
    """Build a control-ready epJSON from a raw dataset epJSON.

    The dataset already ships epJSON v25.1.0 so no upgrade/conversion is
    needed — we only add meters, set the timestep, and make controllable.
    """
    current: Derivation = ExtractFromZip(root_zip, epjson_filename)
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)
    current = modify_timestep(current, timesteps_per_hour=timesteps_per_hour)
    run_period = TaskConfig.from_dict({"run_period": run_period_name}).run_period
    current = modify_run_period(
        current,
        begin_day_of_month=run_period.begin_day_of_month,
        begin_month=run_period.begin_month,
        end_day_of_month=run_period.end_day_of_month,
        end_month=run_period.end_month,
    )
    current = Rename("building.epjson", current)
    return make_controllable(current)


def search_buildings(
    building_type: BuildingType | None = None,
    place: str | None = None,
    building_id: int | None = None,
    run_period: str = "full_year",
    timesteps_per_hour: int = 12,
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
        return lambda: _build_control_derivation(
            root_zip,
            epjson_filename,
            run_period_name=run_period,
            timesteps_per_hour=timesteps_per_hour,
        )

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

    bldg_section = cfg.get("bldg", {})
    if not isinstance(bldg_section, dict):
        bldg_section = {}
    bldg_query: dict[str, Any] = {}
    for k in ("building_type", "place", "building_id"):
        if k in bldg_section:
            bldg_query[k] = bldg_section[k]

    task_section = cfg.get("task", {}) if isinstance(cfg, dict) else {}
    if not isinstance(task_section, dict):
        task_section = {}
    task_config = TaskConfig.from_dict(task_section)

    expose_heating_only_zones = bool(cfg.get("expose_heating_only_zones", True))

    rows = search_buildings(
        run_period=task_config.run_period.name,
        timesteps_per_hour=task_config.timesteps_per_hour,
        **bldg_query,
    )
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
            epw_derivation = ExtractFromZip(root_zip, row.weather_file)

            control_derivation = row.derivation_thunk()
            epjson, hvac_equipment = realize(STORE_PATH.get(), control_derivation)

            metadata = realize(
                STORE_PATH.get(),
                extract_discovery_metadata(Constant(epjson), epw_derivation),
            )

            epw = realize(STORE_PATH.get(), epw_derivation)

            reward_section = cfg.get("reward", {}) if isinstance(cfg, dict) else {}
            if (
                not isinstance(reward_section, dict)
                or "reward_type" not in reward_section
            ):
                raise ValueError(
                    "The 'reward' section with a 'reward_type' key is required "
                    "in the building config."
                )
            reward_config = reward_config_from_dict(reward_section)

            configs.append(
                BuildingConfig(
                    path_to_building=epjson,
                    path_to_weather=epw,
                    reward_config=reward_config,
                    hvac_equipment=hvac_equipment,
                    eplus_output_dir=eplus_output_dir,
                    warmup_phases=metadata.warmup_phases,
                    area=metadata.net_conditioned_area,
                    source_metadata=source_meta,
                    task_config=task_config,
                    expose_heating_only_zones=expose_heating_only_zones,
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
                logger.exception(
                    "Failed to write pipeline error record to %s", err_path
                )

            logger.warning(
                "Failed to build BuildingConfig for building_id=%s: %s",
                source_meta.get("building_id", "unknown"),
                e,
            )
            continue

    return configs


def search_config(
    config: dict | object | None = None,
    eplus_output_dir: Path = Path("eplus_out"),
) -> BuildingConfig:
    configs = search_configs(config=config, n=1, eplus_output_dir=eplus_output_dir)
    if not configs:
        raise RuntimeError("No multizones building configuration found")
    return configs[0]
