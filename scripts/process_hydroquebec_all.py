"""
Process every building in the Hydro-Québec dataset and record summary metadata.

For each building, we store:
- building id
- number of conditioned zones and unconditioned zones
- actuator names in the action space
- success flag, and a short error message on failure

Outputs are written under `outputs/` in the repo root.

IMPORTANT: This script forces the derivation store (downloads / conversions /
prepared epJSONs) to live under $SCRATCH, not in the home directory.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildingProcessingRecord:
    building_id: int | None
    dataset_row_index: int | None
    n_conditioned_zones: int
    n_unconditioned_zones: int
    actuator_names: list[str]
    success: bool
    error: str | None


_IDF_ID_RE = re.compile(r"IDFsAndSchedules/(\d+)/in\.idf$")


def _parse_building_id_from_idf_filename(idf_filename: object) -> int | None:
    if not isinstance(idf_filename, str):
        return None
    m = _IDF_ID_RE.search(idf_filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _ensure_outputs_dir(repo_root: Path) -> Path:
    out = repo_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _require_scratch_store() -> Path:
    scratch_raw = os.environ.get("SCRATCH")
    if not scratch_raw:
        raise RuntimeError(
            "SCRATCH is not set. Refusing to run because this script must cache "
            "processed files in $SCRATCH (not in $HOME)."
        )

    scratch = Path(scratch_raw).expanduser().resolve()
    store_path = (scratch / "building2building").resolve()

    home = Path.home().expanduser().resolve()
    if store_path.is_relative_to(home):
        raise RuntimeError(
            f"Refusing to use STORE_PATH={store_path} because it is under home={home}."
        )

    # Ensure all building2building caching goes here.
    os.environ["STORE_PATH"] = str(store_path)
    return store_path


def _load_epjson(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"epJSON root must be a dict, got {type(obj).__name__}")
    return obj


def _zone_names_from_zonelist(epjson: dict[str, Any], zonelist_name: str) -> set[str]:
    out: set[str] = set()
    all_zones = (
        set(epjson.get("Zone", {}).keys()) if isinstance(epjson.get("Zone"), dict) else set()
    )
    zonelists = epjson.get("ZoneList", {})
    if not isinstance(zonelists, dict):
        return out
    zl = zonelists.get(zonelist_name)
    if not isinstance(zl, dict):
        return out

    # epJSON ZoneList fields are typically zone_1_name, zone_2_name, ...
    for k, v in zl.items():
        if not isinstance(k, str):
            continue
        if not k.lower().endswith("_name"):
            continue
        if isinstance(v, str) and v in all_zones:
            out.add(v)
    return out


def _get_conditioned_and_unconditioned_zone_counts(epjson: dict[str, Any]) -> tuple[int, int]:
    zones_obj = epjson.get("Zone", {})
    if not isinstance(zones_obj, dict):
        all_zones: set[str] = set()
    else:
        all_zones = set(map(str, zones_obj.keys()))

    conditioned: set[str] = set()

    # Most reliable: zones that have HVAC equipment connections.
    conns = epjson.get("ZoneHVAC:EquipmentConnections", {})
    if isinstance(conns, dict):
        for _name, conn in conns.items():
            if not isinstance(conn, dict):
                continue
            z = conn.get("zone_name")
            if isinstance(z, str) and z in all_zones:
                conditioned.add(z)

    # Additional signal: zones (or ZoneLists) referenced by thermostats.
    thermostats = epjson.get("ZoneControl:Thermostat", {})
    if isinstance(thermostats, dict):
        for _name, ctrl in thermostats.items():
            if not isinstance(ctrl, dict):
                continue
            ref = ctrl.get("zone_or_zonelist_name")
            if isinstance(ref, str):
                if ref in all_zones:
                    conditioned.add(ref)
                else:
                    conditioned.update(_zone_names_from_zonelist(epjson, ref))

    unconditioned = all_zones - conditioned
    return len(conditioned), len(unconditioned)


def _short_error(e: BaseException) -> str:
    msg = str(e).strip().replace("\n", " ")
    if len(msg) > 240:
        msg = msg[:240] + "…"
    return f"{type(e).__name__}: {msg}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process every building in the Hydro-Québec dataset."
    )
    p.add_argument(
        "--dataset-zip",
        type=str,
        default="big",
        choices=["big", "small"],
        help=(
            "Which Hydro-Québec dataset zip to use. "
            "'big' downloads the full dataset; 'small' uses the bundled test zip."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/hydroquebec_processing_results.json",
        help="Where to write the JSON summary (relative to repo root).",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the dataset rows (0-based).",
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop index in the dataset rows (0-based, exclusive).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N buildings (applied after start/stop slicing).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Rewrite the JSON output every N buildings to avoid losing progress.",
    )
    return p.parse_args()


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _pick_hq_csv_name(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

    # Prefer the known big-dataset file name if present.
    preferred = "2026-01-29_building-stock-10000-mila.csv"
    if preferred in names:
        return preferred

    csvs = [n for n in names if n.lower().endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"No CSV files found in {zip_path}")

    # Heuristic: pick the most likely building stock CSV.
    def score(name: str) -> tuple[int, int]:
        low = name.lower()
        return (
            int("building" in low and "stock" in low),
            int("mila" in low),
        )

    csvs_sorted = sorted(csvs, key=lambda n: (score(n), n), reverse=True)
    return csvs_sorted[0]


def _prefix_weather_dir(v: str) -> str:
    s = v.strip()
    if s.startswith("weather/"):
        return s
    return f"weather/{s}"


def _load_hq_dataframe_from_zip(zip_path: Path) -> "Any":
    # Returns a pandas DataFrame (duckdb -> pandas), but keep it untyped here to
    # avoid importing pandas at module import time.
    import duckdb

    csv_name = _pick_hq_csv_name(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        csv_text = z.open(csv_name).read().decode("utf-8")
        mapping = None
        if "Mapping-Region-EPWfiles.json" in z.namelist():
            mapping = json.loads(z.read("Mapping-Region-EPWfiles.json"))

    df = duckdb.from_csv_auto(io.StringIO(csv_text)).to_df()

    if "Region_Administrative" in df.columns and isinstance(mapping, dict):
        def _region_to_epw(region: object) -> str:
            if not isinstance(region, str):
                raise TypeError(f"Region_Administrative must be a string, got {region!r}")
            epw = mapping.get(region)
            if not isinstance(epw, str):
                raise KeyError(f"No EPW mapping for Region_Administrative={region!r}")
            return _prefix_weather_dir(epw)

        df = df.assign(epw_filename=df["Region_Administrative"].apply(_region_to_epw))
    elif "weather_station_epw_filepath" in df.columns:
        df = df.assign(
            epw_filename=df["weather_station_epw_filepath"].apply(
                lambda name: _prefix_weather_dir(str(name))
            )
        )
    elif "epw_filename" in df.columns:
        df = df.assign(epw_filename=df["epw_filename"].apply(lambda name: _prefix_weather_dir(str(name))))
    else:
        raise KeyError(
            "Could not derive EPW filename: expected one of "
            "['weather_station_epw_filepath', 'Region_Administrative', 'epw_filename'] in dataset CSV."
        )

    df = df.assign(idf_filename=[f"IDFsAndSchedules/{i}/in.idf" for i in range(1, len(df) + 1)])
    df = df.assign(
        schedule_filename=[f"IDFsAndSchedules/{i}/in.schedules.csv" for i in range(1, len(df) + 1)]
    )
    return df


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_outputs_dir(repo_root)

    store_path = _require_scratch_store()

    # Import after STORE_PATH is set to avoid ever defaulting to $HOME.
    from building2building.env import STORE_PATH  # noqa: WPS433
    from building2building.env import energyplus_path  # noqa: WPS433
    from building2building.sources import hydroquebec  # noqa: WPS433
    from building2building.store import realize  # noqa: WPS433

    STORE_PATH.set(store_path)

    # Enumerate all buildings (duckdb->pandas). Each row has a callable that yields
    # the derivation graph for the building.
    root_zip = (
        hydroquebec.dataset_zip_small()
        if args.dataset_zip == "small"
        else hydroquebec.dataset_zip()
    )
    zip_path = Path(realize(STORE_PATH.get(), root_zip))
    df = _load_hq_dataframe_from_zip(zip_path)

    # Add derivation thunk, mirroring hydroquebec.search_buildings().
    ep = energyplus_path()

    def trans(idf_filename: str, schedule_filename: str):
        return lambda: hydroquebec._build_control_derivation(  # noqa: SLF001
            root_zip, idf_filename, schedule_filename, ep
        )

    df = df.assign(
        derivation_thunk=list(map(trans, df.idf_filename, df.schedule_filename))
    )
    df_sliced = df.iloc[args.start : args.stop]  # type: ignore[misc]
    if args.limit is not None:
        df_sliced = df_sliced.iloc[: args.limit]  # type: ignore[misc]

    output_path = (repo_root / args.output).resolve()

    records: list[BuildingProcessingRecord] = []
    processed = 0

    for row_idx, row in df_sliced.iterrows():
        building_id = _parse_building_id_from_idf_filename(row.get("idf_filename"))
        dataset_row_index = int(row_idx) if row_idx is not None else None

        try:
            thunk = row.get("derivation_thunk")
            if not callable(thunk):
                raise TypeError("row.derivation_thunk is missing or not callable")

            epjson_path, actuator_descs = realize(STORE_PATH.get(), thunk())

            epjson = _load_epjson(Path(epjson_path))
            n_cond, n_uncond = _get_conditioned_and_unconditioned_zone_counts(epjson)

            actuator_names = []
            for a in actuator_descs:
                try:
                    actuator_names.append(str(a.component_name))
                except Exception:
                    actuator_names.append(str(a))

            rec = BuildingProcessingRecord(
                building_id=building_id,
                dataset_row_index=dataset_row_index,
                n_conditioned_zones=n_cond,
                n_unconditioned_zones=n_uncond,
                actuator_names=actuator_names,
                success=True,
                error=None,
            )
        except Exception as e:
            rec = BuildingProcessingRecord(
                building_id=building_id,
                dataset_row_index=dataset_row_index,
                n_conditioned_zones=0,
                n_unconditioned_zones=0,
                actuator_names=[],
                success=False,
                error=_short_error(e),
            )

        records.append(rec)
        processed += 1

        if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
            _atomic_write_json(output_path, [r.__dict__ for r in records])

    _atomic_write_json(output_path, [r.__dict__ for r in records])


if __name__ == "__main__":
    main()

