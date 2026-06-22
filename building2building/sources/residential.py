import io
import itertools
import json
import logging
import subprocess
import tempfile
import traceback
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import List

import duckdb
from building2building import pipeline
from building2building.env import STORE_PATH, energyplus_path

# Extract metadata from control epJSON
from building2building.pipeline import (
    create_complete_pipeline,
    extract_discovery_metadata,
    link_in_schedule,
    make_controllable,
    modify_run_period,
    prepare_building,
)
from building2building.store import (
    OUTPUT,
    Constant,
    Derivation,
    DownloadFile,
    ExtractFromZip,
    ExtractZip,
    LocalFile,
    Realizable,
    derivation,
    realize,
)
from building2building.types import (
    BuildingConfig,
    TaskConfig,
    reward_config_from_dict,
)
from pandas import DataFrame

logger = logging.getLogger(__name__)


def _row_source_metadata(row) -> dict[str, object]:
    """
    Extract a compact, JSON-friendly subset of identifying info from the selected row.
    Use for logging to identify the chosen building.
    """
    meta: dict[str, object] = {"source": "residential"}
    # DataFrame index from duckdb/parquet (helps uniquely identify the chosen row)
    try:
        meta["dataset_row_index"] = int(row.name)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Useful building descriptors (only if present)
    for k in (
        "geometry_unit_type",
        "geometry_building_num_units",
        "year_built",
        "weather_station_epw_filepath",
        "Region_Administrative",
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
def dataset_zip_small() -> Derivation:
    place = files("building2building.sources.data") / "hydroquebec.zip"
    if not isinstance(place, Path):
        raise Exception("error")

    return LocalFile(place)


def dataset_zip() -> Derivation:
    return DownloadFile(
        "hydroquebec_big.zip",
        "https://huggingface.co/datasets/Terramorpha/b2b-hq-big/resolve/main/hydroquebec_big.zip",
        bytes.fromhex(
            "ce6f77015250a959fb76869e158f275ce5de286779b7d0818e190ab72c54a3d8"
        ),
    )


@derivation("table.parquet")
def table_index(root_zip: Path):
    out = OUTPUT.get()

    with zipfile.ZipFile(root_zip) as zip_ref:
        contents = io.StringIO(
            zip_ref.open("2026-01-29_building-stock-10000-mila.csv")
            .read()
            .decode("utf-8")
        )

    df = duckdb.from_csv_auto(contents).to_df()

    # Dataset schema differs slightly between zip variants. derive EPW from the
    # administrative region mapping shipped in the zip. Otherwise use the
    # `weather_station_epw_filepath` if present.
    if "Region_Administrative" in df.columns:
        with zipfile.ZipFile(root_zip) as zip_ref:
            mapping = json.loads(zip_ref.read("Mapping-Region-EPWfiles.json"))
        if not isinstance(mapping, dict):
            raise TypeError(
                "Expected Mapping-Region-EPWfiles.json to be a JSON object (dict)."
            )

        def _region_to_epw(region: object) -> str:
            if not isinstance(region, str):
                raise TypeError(
                    f"Region_Administrative must be a string, got {region!r}"
                )
            epw = mapping.get(region)
            if not isinstance(epw, str):
                raise KeyError(f"No EPW mapping for Region_Administrative={region!r}")
            return f"weather/{epw}"

        df = df.assign(epw_filename=df["Region_Administrative"].apply(_region_to_epw))
    elif "weather_station_epw_filepath" in df.columns:
        df = df.assign(
            epw_filename=df["weather_station_epw_filepath"].apply(
                lambda name: f"weather/{name}"
            )
        )
    else:
        raise KeyError(
            "Could not derive EPW filename: expected one of "
            "['weather_station_epw_filepath', 'Region_Administrative'] in dataset CSV."
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
    root_zip: Realizable,
    idf_filename: str,
    schedule_filename: str,
    ep: Realizable,
    controls: list[str],
    run_period_name: str,
    timesteps_per_hour: int = 12,
):
    """
    Build control-ready epJSON from IDF (residential-specific).

    Pipeline: IDF → prepare → link schedule → make controllable
    """

    idf_derivation = ExtractFromZip(root_zip, idf_filename)

    # Step 1: Prepare epJSON (upgrade, convert, add meters, set timestep)
    epjson = prepare_building(
        idf_derivation,
        ep,
        src_version="24.2.0",
        timesteps_per_hour=timesteps_per_hour,
    )
    run_period = TaskConfig.from_dict({"run_period": run_period_name}).run_period
    epjson = modify_run_period(
        epjson,
        begin_day_of_month=run_period.begin_day_of_month,
        begin_month=run_period.begin_month,
        end_day_of_month=run_period.end_day_of_month,
        end_month=run_period.end_month,
    )

    schedule_derivation = ExtractFromZip(root_zip, schedule_filename)
    # Step 2: Link schedule data (residential-specific)
    epjson = link_in_schedule(epjson, schedule_derivation)

    # Step 3: Make controllable
    return make_controllable(epjson, controls=controls)


def search_buildings(
    run_period: str = "full_year",
    timesteps_per_hour: int = 12,
    **query,
) -> DataFrame:
    root_zip = dataset_zip()
    index = realize(STORE_PATH.get(), table_index(root_zip))
    ep = energyplus_path()
    controls = query.get("controls")

    def trans(idf_filename, schedule_filename):
        return lambda: _build_control_derivation(
            root_zip,
            idf_filename,
            schedule_filename,
            ep,
            controls,
            run_period_name=run_period,
            timesteps_per_hour=timesteps_per_hour,
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

    bldg_section = cfg.get("bldg", {})
    if not isinstance(bldg_section, dict):
        bldg_section = {}
    config_nn = bldg_section.get("query", {})
    if not isinstance(config_nn, dict):
        config_nn = {}

    task_section = cfg.get("task", {}) if isinstance(cfg, dict) else {}
    if not isinstance(task_section, dict):
        task_section = {}
    task_config = TaskConfig.from_dict(task_section)

    rows = search_buildings(
        run_period=task_config.run_period.name,
        timesteps_per_hour=task_config.timesteps_per_hour,
        **config_nn,
    )
    # Heuristic: when no explicit filters are provided, prioritize simpler
    # buildings first to avoid long sequences of E+ fatals during discovery.
    try:
        if "geometry_unit_type" in rows.columns:
            rows = (
                rows.assign(
                    _b2b_priority=rows["geometry_unit_type"]
                    .astype(str)
                    .ne("single-family detached")
                )
                .sort_values("_b2b_priority", kind="stable")
                .drop(columns=["_b2b_priority"])
            )
    except Exception:
        # Best-effort only; never fail selection due to prioritization.
        pass

    root_zip = dataset_zip()

    configs: list[BuildingConfig] = []
    # Try rows until we successfully build `n` configs. Some buildings can fail
    # the discovery simulation (E+ fatal) and we don't want that to yield an
    # empty result when `n=1`.
    for _, row in rows.iterrows():
        if len(configs) >= n:
            break
        source_meta = _row_source_metadata(row)
        try:
            epw_derivation = ExtractFromZip(
                root_zip, row.epw_filename
            )  # use the weather file for THIS building

            # Get control-ready building with actuators from make_controllable()
            control_derivation = row.derivation_thunk()
            epjson, hvac_equipment = realize(STORE_PATH.get(), control_derivation)

            metadata = realize(
                STORE_PATH.get(),
                extract_discovery_metadata(Constant(epjson), epw_derivation),
            )

            epw = realize(STORE_PATH.get(), epw_derivation)

            area = metadata.net_conditioned_area
            warmup_phases = metadata.warmup_phases

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
                    warmup_phases=warmup_phases,
                    area=area,
                    source_metadata=source_meta,
                    task_config=task_config,
                )
            )
        except Exception as e:
            # Critical for large-scale runs: log and continue so bad buildings are traceable.
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
                # If even this fails, at least surface the info in logs.
                logger.exception(
                    "Failed to write pipeline error record to %s", err_path
                )

            logger.warning(
                "Failed to build BuildingConfig for row=%s: %s",
                source_meta.get("dataset_row_index", "unknown"),
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
        raise RuntimeError("No residential building configuration found")
    return configs[0]
