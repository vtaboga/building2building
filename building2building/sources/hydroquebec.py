from importlib.resources import files
from pathlib import Path
import json
import subprocess
import tempfile
import logging
import re 

import duckdb
from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import create_complete_pipeline, link_in_schedule
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
from building2building.utils import get_net_conditioned_area, get_warmup_days

logger = logging.getLogger(__name__)


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
                str(root / "IDFsAndSchedules" / str(i) / "in.idf")
                for i in range(1, len(df) + 1)
            ]
        )

        df = df.assign(
            schedule_path=[
                str(root / "IDFsAndSchedules" / str(i) / "in.schedules.csv")
                for i in range(1, len(df) + 1)
            ]
        )

        df = df.drop(columns=["geometry_roof_pitch"])

        df.to_parquet(str(out))

    return build_database(extracted())


def search_weathers() -> DataFrame:
    index = realize(STORE_PATH.get(), table_index())
    return duckdb.from_parquet(str(index)).to_df()


def search_buildings(config: dict | None = None, n: int = 2) -> DataFrame:
    """
    - If config is None: return full dataset.
    - If config is provided: filter by string keys (exact, case-insensitive),
      then find the n closest matches by lexicographic distance using the
      numeric keys IN THE ORDER they appear in `config`.
    """

    index = realize(STORE_PATH.get(), table_index())
    ep = energyplus_path()

    def trans(idf_path):
        return lambda: create_complete_pipeline(
            Constant(Path(idf_path)), ep, src_version="24.2.0"
        )

    if not config:
        df = duckdb.from_parquet(str(index)).to_df()
        return df.assign(derivation_thunk=df.idf_path.apply(trans))

    # Preserve insertion order from `config`
    string_filters: list[tuple[str, str]] = []
    numeric_order: list[tuple[str, float]] = []
    for k, v in config.items():
        if isinstance(v, str):
            string_filters.append((k, v))
        elif isinstance(v, (int, float)):
            numeric_order.append((k, float(v)))

    # WHERE for string filters
    where_sql = " AND ".join([f'lower("{k}") = lower(?)' for k, _ in string_filters]) or "TRUE"

    if numeric_order:
        # Build distance columns d1, d2, ... in the SAME order as provided
        dist_cols = []
        for i, (k, _) in enumerate(numeric_order, start=1):
            dist_cols.append(
                f'CASE WHEN "{k}" IS NULL THEN 1e12 ELSE ABS(CAST("{k}" AS DOUBLE) - ?) END AS d{i}'
            )
        dist_select = ",\n          ".join(dist_cols)
        order_by = ", ".join([f"d{i}" for i in range(1, len(numeric_order) + 1)])

        sql = f"""
        WITH s AS (
          SELECT * FROM read_parquet(?)
          WHERE {where_sql}
        )
        SELECT
          s.*,
          {dist_select}
        FROM s
        ORDER BY {order_by}
        LIMIT ?
        """
        params: list[object] = [str(index)]
        params += [v for _, v in string_filters]     # string placeholders
        params += [v for _, v in numeric_order]      # numeric targets (for d1, d2, ...)
        params.append(int(n))
    else:
        # No numeric keys: just filter by strings and take first n
        sql = f"""
        SELECT * FROM read_parquet(?)
        WHERE {where_sql}
        LIMIT ?
        """
        params = [str(index)]
        params += [v for _, v in string_filters]
        params.append(int(n))

    con = duckdb.connect()
    df = con.execute(sql, params).df()
    df = df.assign(derivation_thunk=df.idf_path.apply(trans))

    return df
        


def search_building_configs(config: dict | None = None, n: int = 2, eplus_output_dir: Path = Path("eplus_out")) -> list[BuildingConfig]:
    """
    Return a BuildingConfig per selected building, using each row's weather_path.
    """
    rows = search_buildings(config, n)
    configs: list[BuildingConfig] = []
    for _, row in rows.iterrows():
        epw = Path(row.epw_path)  # use the weather file for THIS building
        derivation = link_in_schedule(row.derivation_thunk(), Path(row.schedule_path))
        epjson = realize(STORE_PATH.get(), derivation)

        area, warmup_days = _ensure_metrics_cached(epjson, epw)
        area = area if area is not None else 1.0
        warmup_phases = warmup_days if warmup_days is not None else 1

        configs.append(
            BuildingConfig(
                path_to_building=epjson,
                path_to_weather=epw,
                reward_config=BaseRewardConfig(area, 1.0),
                eplus_output_dir=eplus_output_dir,
                warmup_phases=1,  # keep consistent with existing search_config
            )
        )

    if not configs:
        raise ValueError("No building matched the provided configuration.")

    logger.info(f"Found {len(configs)} building configs matching {config}")

    return configs


def _metrics_cache_path() -> Path:
    cache_dir = STORE_PATH.get()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "hydroquebec_metrics.json"


def _load_cache() -> dict:
    p = _metrics_cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(d: dict) -> None:
    p = _metrics_cache_path()
    try:
        p.write_text(json.dumps(d))
    except Exception:
        pass


def _ensure_metrics_cached(epjson: Path, epw: Path) -> tuple[float | None, float | None]:
    key = f"{epjson.resolve()}|{epw.resolve()}"
    cache = _load_cache()
    if key in cache:
        entry = cache[key]
        area, wdays = entry.get("area"), entry.get("warmup_days")
        logger.info(
            "Using cached metrics: area=%s m², warmup_days=%s (building=%s, weather=%s)",
            area,
            wdays,
            epjson.name,
            epw.name,
        )
        return area, wdays

    area, warmup_days = _compute_metrics_once(epjson, epw)
    cache[key] = {"area": area, "warmup_days": warmup_days}
    _save_cache(cache)
    logger.info(
        "Cached metrics: area=%s m², warmup_days=%s (building=%s, weather=%s)",
        area,
        warmup_days,
        epjson.name,
        epw.name,
    )
    return area, warmup_days


def _compute_metrics_once(epjson: Path, epw: Path) -> tuple[float | None, float | None]:
    ep_root = Path(realize(STORE_PATH.get(), energyplus_path()))
    ep_bin = ep_root / "energyplus"

    with tempfile.TemporaryDirectory(dir=str(STORE_PATH.get())) as tmp:
        out = Path(tmp)
        # Run energyplus without python
        cmd = [str(ep_bin), "-w", str(epw), "-d", str(out), "-x", str(epjson)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return None, None

        # Prefer the summary tables file explicitly (contains WarmupDays table)
        preferred = out / "eplustbl.htm"
        if preferred.exists():
            html_path = preferred
        else:
            # Fallbacks if needed
            candidates = [
                out / "eplusout.htm",
                out / "eplusout.html",
                out / "eplusbl.htm",
            ]
            html_path = next((p for p in candidates if p.exists()), None)
        # Log outputs for debugging
        if not html_path:
            return None, None

        area = get_net_conditioned_area(str(html_path))
        warmup = get_warmup_days(str(html_path))
        return area, warmup
