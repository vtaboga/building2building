import io
import itertools
import json
import logging
import subprocess
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path

import duckdb
from building2building import pipeline
from building2building.env import STORE_PATH, energyplus_path

# Extract metadata from control epJSON
from building2building.pipeline import (
    create_complete_pipeline,
    extract_discovery_metadata,
    link_in_schedule,
    make_controllable,
    prepare_building,
)
from building2building.store import (
    OUTPUT,
    Constant,
    Derivation,
    ExtractFromZip,
    ExtractZip,
    LocalFile,
    Realizable,
    derivation,
    realize,
)
from building2building.types import (
    BarrierRewardConfig,
    BaseRewardConfig,
    BuildingConfig,
    DeadbandRewardConfig,
)
from pandas import DataFrame

logger = logging.getLogger(__name__)

def _row_source_metadata(row) -> dict[str, object]:
    """
    Extract a compact, JSON-friendly subset of identifying info from the selected row.
    Use for logging to identify the chosen building.
    """
    meta: dict[str, object] = {"source": "hydroquebec"}
    # DataFrame index from duckdb/parquet (helps uniquely identify the chosen row)
    try:
        meta["dataset_row_index"] = int(row.name)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Common identifiers we know we add in `table_index`
    for k in ("idf_filename", "schedule_filename", "epw_filename"):
        try:
            if k in row and row[k] is not None:
                meta[k] = str(row[k])
        except Exception:
            continue

    # Useful building descriptors (only if present)
    for k in (
        "geometry_unit_type",
        "geometry_building_num_units",
        "year_built",
        "weather_station_epw_filepath",
    ):
        try:
            if k in row and row[k] is not None:
                v = row[k]
                if isinstance(v, (int, float, str, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
        except Exception:
            continue

    return meta


# We should try not to call this function too often. Each call of LocalFile
# requires reading the file in its entirety, which is bad. Perhaps this should
# use LocalSymlink?
def dataset_zip() -> Derivation:
    place = files("building2building.sources.data") / "hydroquebec.zip"
    if not isinstance(place, Path):
        raise Exception("error")

    return LocalFile(place)


@derivation("table.parquet")
def table_index(root_zip: Path):
    out = OUTPUT.get()

    with zipfile.ZipFile(root_zip) as zip_ref:
        contents = io.StringIO(
            zip_ref.open("2025-10-09_building-stock-100-mila.csv")
            .read()
            .decode("utf-8")
        )

    df = duckdb.from_csv_auto(contents).to_df()

    df = df.assign(
        epw_filename=df.weather_station_epw_filepath.apply(
            lambda name: f"weather/{name}"
        )
    )

    df = df.assign(
        idf_filename=[f"IDFsAndSchedules/{i}/in.idf" for i in range(1, len(df) + 1)]
    )

    df = df.assign(
        schedule_filename=[
            f"IDFsAndSchedules/{i}/in.schedules.csv" for i in range(1, len(df) + 1)
        ]
    )

    df = df.drop(columns=["geometry_roof_pitch"])

    df.to_parquet(str(out))


def _build_control_derivation(
    root_zip: Realizable, idf_filename: str, schedule_filename: str, ep: Realizable
):
    """
    Build control-ready epJSON from IDF (hydroquebec-specific).

    Pipeline: IDF → prepare → link schedule → make controllable
    """

    idf_derivation = ExtractFromZip(root_zip, idf_filename)

    # Step 1: Prepare epJSON (upgrade, convert, add meters, set timestep)
    epjson = prepare_building(idf_derivation, ep, src_version="24.2.0")

    schedule_derivation = ExtractFromZip(root_zip, schedule_filename)
    # Step 2: Link schedule data (hydroquebec-specific)
    epjson = link_in_schedule(epjson, schedule_derivation)

    # Step 3: Make controllable
    return make_controllable(epjson)


def search_buildings(**query) -> DataFrame:
    root_zip = dataset_zip()
    index = realize(STORE_PATH.get(), table_index(root_zip))
    ep = energyplus_path()

    def trans(idf_filename, schedule_filename):
        return lambda: _build_control_derivation(
            root_zip, idf_filename, schedule_filename, ep
        )

    db = duckdb.from_parquet(str(index))

    for k, v in query.items():
        if isinstance(v, str):
            # Case-insensitive exact match on strings.
            db = db.filter(
                duckdb.FunctionExpression("lower", duckdb.ColumnExpression(k))
                == duckdb.ConstantExpression(v.lower())
            )
        elif isinstance(v, (int, float)):
            # Prefer closest numeric match (e.g., year_built).
            # NOTE: duckdb relations are immutable; `order()` returns a new relation.
            db = db.order(f"abs({k} - {v})")

    df = db.to_df()

    df = df.assign(
        derivation_thunk=list(map(trans, df.idf_filename, df.schedule_filename))
    )
    return df


def search_configs(
    config: dict | object | None = None,
    n: int = 2,
    eplus_output_dir: Path = Path("eplus_out"),
) -> list[BuildingConfig]:
    """
    Return a BuildingConfig per selected building, using each row's weather_path.
    """
    cfg_any = config or {}
    if not isinstance(cfg_any, dict):
        # Hydra passes OmegaConf objects; convert to plain dict so `.get()` and
        # `isinstance(..., dict)` logic behaves as expected.
        try:
            from omegaconf import OmegaConf  # type: ignore

            cfg = OmegaConf.to_container(cfg_any, resolve=True)  # type: ignore[assignment]
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception:
            cfg = {}
    else:
        cfg = cfg_any

    config_nn = cfg.get("bldg", {})
    # Our bldg group configs are nested like: bldg: { bldg: {...} }
    if (
        isinstance(config_nn, dict)
        and "bldg" in config_nn
        and isinstance(config_nn["bldg"], dict)
    ):
        config_nn = config_nn["bldg"]

    rows = search_buildings(**config_nn)

    root_zip = dataset_zip()

    configs: list[BuildingConfig] = []
    for _, row in itertools.islice(rows.iterrows(), n):
        epw_derivation = ExtractFromZip(
            root_zip, row.epw_filename
        )  # use the weather file for THIS building

        # Get control-ready building with actuators from make_controllable()
        control_derivation = row.derivation_thunk()
        epjson, actuator_descriptions = realize(STORE_PATH.get(), control_derivation)
        metadata = realize(
            STORE_PATH.get(),
            extract_discovery_metadata(Constant(epjson), epw_derivation),
        )

        epw = realize(STORE_PATH.get(), epw_derivation)

        area = metadata.net_conditioned_area
        warmup_phases = metadata.warmup_phases

        reward_section = cfg.get("reward", {}) if isinstance(cfg, dict) else {}
        if not isinstance(reward_section, dict):
            reward_section = {}
        reward_type = reward_section.get("reward_type")

        if reward_type == "DeadbandRewardConfig":
            energy_weight = reward_section.get("energy_weight")
            target_temp = reward_section.get("target_temp")
            dT = reward_section.get("dT")
            reward_config = DeadbandRewardConfig(
                area=area,
                energy_weight=energy_weight,
                target_temp=target_temp,
                dT=dT,
            )
        elif reward_type == "BaseRewardConfig":
            energy_weight = reward_section.get("energy_weight")
            reward_config = BaseRewardConfig(
                energy_weight=energy_weight,
            )
        elif reward_type == "BarrierRewardConfig":
            energy_weight = reward_section.get("energy_weight")
            reward_config = BarrierRewardConfig(
                energy_weight=energy_weight,
            )
        elif reward_type is None:
            # Back-compat / convenience: allow callers to omit reward config entirely.
            reward_config = BaseRewardConfig(energy_weight=0.0)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

        configs.append(
            BuildingConfig(
                path_to_building=epjson,
                path_to_weather=epw,
                reward_config=reward_config,
                hvac_actuators=actuator_descriptions,
                eplus_output_dir=eplus_output_dir,
                warmup_phases=warmup_phases,
                area=area,
                source_metadata=_row_source_metadata(row),
            )
        )

    return configs
