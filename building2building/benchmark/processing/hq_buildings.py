"""
Hydro-Québec buildings processing (frozen, benchmark-oriented).

This module provides a stable processing pipeline that:
- enumerates Hydro-Québec buildings from the dataset zip
- converts each IDF to a controllable epJSON
- records per-building summary metadata (zone counts, actuator names, success/errors)

The key backward-compatibility guarantee is that the default controls list is
explicit and intentionally conservative, so newly-added actuators in
`b2b.pipeline.actuators` do not change results unless explicitly enabled.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

ControlName = Literal[
    "unitary_hvac",
    "baseboard",
    "fanonoff",
    "waterheater",
    "pump",
    "airterminal",
    "controller_outdoorair",
]


_IDF_ID_RE = re.compile(r"IDFsAndSchedules/(\d+)/in\.idf$")


@dataclass(frozen=True)
class BuildingProcessingRecord:
    building_id: int | None
    dataset_row_index: int | None
    n_conditioned_zones: int
    n_unconditioned_zones: int
    actuator_names: list[str]
    success: bool
    error: str | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "building_id": self.building_id,
            "dataset_row_index": self.dataset_row_index,
            "n_conditioned_zones": self.n_conditioned_zones,
            "n_unconditioned_zones": self.n_unconditioned_zones,
            "actuator_names": list(self.actuator_names),
            "success": self.success,
            "error": self.error,
        }


@dataclass(frozen=True)
class HQBuildingsProcessingConfig:
    output: str = "hq_buildings_processing_results.json"
    start: int = 0
    stop: int | None = None
    limit: int | None = None
    checkpoint_every: int = 100
    # IMPORTANT: keep this default list stable for backward compatibility.
    controls: tuple[ControlName, ...] = ("unitary_hvac", "baseboard", "fanonoff")
    # If set, must be under $SCRATCH and not under $HOME.
    store_path: str | None = None
    dry_run: bool = False

    @classmethod
    def from_cfg(cls, cfg: object) -> "HQBuildingsProcessingConfig":
        """
        Build from a Hydra DictConfig or a plain mapping.

        Expected shape:
          processing:
            dataset_zip: big|small
            output: <filename or path>
            start: int
            stop: int|null
            limit: int|null
            checkpoint_every: int
            controls: [..]
            store_path: str|null
            dry_run: bool
        """
        data = _to_plain_dict(cfg)
        processing = data.get("processing", {})
        if not isinstance(processing, dict):
            raise TypeError("cfg.processing must be a mapping")

        output_raw = processing.get("output", cls.output)
        if not isinstance(output_raw, str) or not output_raw.strip():
            raise TypeError("processing.output must be a non-empty string")

        start_raw = processing.get("start", cls.start)
        if not isinstance(start_raw, int) or start_raw < 0:
            raise TypeError("processing.start must be an int >= 0")

        stop_raw = processing.get("stop", cls.stop)
        if stop_raw is not None and (not isinstance(stop_raw, int) or stop_raw < 0):
            raise TypeError("processing.stop must be null or an int >= 0")

        limit_raw = processing.get("limit", cls.limit)
        if limit_raw is not None and (not isinstance(limit_raw, int) or limit_raw < 0):
            raise TypeError("processing.limit must be null or an int >= 0")

        checkpoint_raw = processing.get("checkpoint_every", cls.checkpoint_every)
        if not isinstance(checkpoint_raw, int) or checkpoint_raw < 0:
            raise TypeError("processing.checkpoint_every must be an int >= 0")

        controls_raw = processing.get("controls", list(cls.controls))
        controls = _parse_controls(controls_raw)

        store_path_raw = processing.get("store_path", cls.store_path)
        if store_path_raw is not None and (not isinstance(store_path_raw, str) or not store_path_raw.strip()):
            raise TypeError("processing.store_path must be null or a non-empty string")

        dry_run_raw = processing.get("dry_run", cls.dry_run)
        if not isinstance(dry_run_raw, bool):
            raise TypeError("processing.dry_run must be a bool")

        return cls(
            output=output_raw,
            start=start_raw,
            stop=stop_raw,
            limit=limit_raw,
            checkpoint_every=checkpoint_raw,
            controls=tuple(controls),
            store_path=store_path_raw,
            dry_run=dry_run_raw,
        )


def run_hq_buildings_processing(cfg: object, *, output_dir: Path) -> Path:
    """
    Run processing using configuration from Hydra.

    Writes a JSON list of BuildingProcessingRecord dicts to:
      output_dir / cfg.processing.output (unless output is absolute)
    """
    config = HQBuildingsProcessingConfig.from_cfg(cfg)

    store_path = _require_scratch_store(config.store_path)

    # Import after STORE_PATH is set to avoid any defaulting to $HOME.
    from b2b.env import STORE_PATH, energyplus_path  # noqa: WPS433
    from b2b.pipeline import link_in_schedule, make_controllable, prepare_building  # noqa: WPS433
    from b2b.sources import hydroquebec  # noqa: WPS433
    from b2b.store import ExtractFromZip, Realizable, realize  # noqa: WPS433

    STORE_PATH.set(store_path)

    if config.dry_run:
        out_path = _resolve_output_path(output_dir, config.output)
        _atomic_write_json(out_path, [])
        return out_path

    root_zip = hydroquebec.dataset_zip()
    zip_path = Path(realize(STORE_PATH.get(), root_zip))
    df = _load_hq_dataframe_from_zip(zip_path)

    ep = energyplus_path()

    def _build_control_derivation_frozen_v1(
        root_zip_realizable: Realizable,
        idf_filename: str,
        schedule_filename: str,
        ep_realizable: Realizable,
        *,
        controls: Sequence[ControlName],
    ):
        idf_derivation = ExtractFromZip(root_zip_realizable, idf_filename)
        epjson = prepare_building(idf_derivation, ep_realizable, src_version="24.2.0")
        schedule_derivation = ExtractFromZip(root_zip_realizable, schedule_filename)
        epjson = link_in_schedule(epjson, schedule_derivation)
        return make_controllable(epjson, controls=list(controls))

    def trans(idf_filename: str, schedule_filename: str):
        return lambda: _build_control_derivation_frozen_v1(
            root_zip,
            idf_filename,
            schedule_filename,
            ep,
            controls=config.controls,
        )

    df = df.assign(
        derivation_thunk=list(map(trans, df.idf_filename, df.schedule_filename))
    )

    df_sliced = df.iloc[config.start : config.stop]  # type: ignore[misc]
    if config.limit is not None:
        df_sliced = df_sliced.iloc[: config.limit]  # type: ignore[misc]

    output_path = _resolve_output_path(output_dir, config.output)

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

            actuator_names: list[str] = []
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

        if config.checkpoint_every > 0 and processed % config.checkpoint_every == 0:
            _atomic_write_json(output_path, [r.to_json_dict() for r in records])

    _atomic_write_json(output_path, [r.to_json_dict() for r in records])
    return output_path


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


def _require_scratch_store(store_path_raw: str | None) -> Path:
    scratch_raw = os.environ.get("SCRATCH")
    if not scratch_raw:
        raise RuntimeError(
            "SCRATCH is not set. Refusing to run because processing must cache "
            "processed files in $SCRATCH (not in $HOME)."
        )

    scratch = Path(scratch_raw).expanduser().resolve()

    if store_path_raw is not None:
        store_path = Path(store_path_raw).expanduser().resolve()
    else:
        env_store_path = os.environ.get("STORE_PATH")
        store_path = (
            Path(env_store_path).expanduser().resolve()
            if env_store_path
            else (scratch / "Building2Building" / "store").resolve()
        )

    home = Path.home().expanduser().resolve()
    if store_path.is_relative_to(home):
        raise RuntimeError(
            f"Refusing to use STORE_PATH={store_path} because it is under home={home}."
        )

    if not store_path.is_relative_to(scratch):
        raise RuntimeError(
            f"Refusing to use STORE_PATH={store_path} because it is not under "
            f"SCRATCH={scratch}."
        )

    store_path.mkdir(parents=True, exist_ok=True)
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
        set(epjson.get("Zone", {}).keys())
        if isinstance(epjson.get("Zone"), dict)
        else set()
    )
    zonelists = epjson.get("ZoneList", {})
    if not isinstance(zonelists, dict):
        return out
    zl = zonelists.get(zonelist_name)
    if not isinstance(zl, dict):
        return out

    for k, v in zl.items():
        if not isinstance(k, str):
            continue
        if not k.lower().endswith("_name"):
            continue
        if isinstance(v, str) and v in all_zones:
            out.add(v)
    return out


def _get_conditioned_and_unconditioned_zone_counts(
    epjson: dict[str, Any],
) -> tuple[int, int]:
    zones_obj = epjson.get("Zone", {})
    if not isinstance(zones_obj, dict):
        all_zones: set[str] = set()
    else:
        all_zones = set(map(str, zones_obj.keys()))

    conditioned: set[str] = set()

    conns = epjson.get("ZoneHVAC:EquipmentConnections", {})
    if isinstance(conns, dict):
        for _name, conn in conns.items():
            if not isinstance(conn, dict):
                continue
            z = conn.get("zone_name")
            if isinstance(z, str) and z in all_zones:
                conditioned.add(z)

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

    preferred = "2026-01-29_building-stock-10000-mila.csv"
    if preferred in names:
        return preferred

    csvs = [n for n in names if n.lower().endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"No CSV files found in {zip_path}")

    def score(name: str) -> tuple[int, int]:
        low = name.lower()
        return (int("building" in low and "stock" in low), int("mila" in low))

    csvs_sorted = sorted(csvs, key=lambda n: (score(n), n), reverse=True)
    return csvs_sorted[0]


def _prefix_weather_dir(v: str) -> str:
    s = v.strip()
    if s.startswith("weather/"):
        return s
    return f"weather/{s}"


def _load_hq_dataframe_from_zip(zip_path: Path) -> "Any":
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
                raise TypeError(
                    f"Region_Administrative must be a string, got {region!r}"
                )
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
        df = df.assign(
            epw_filename=df["epw_filename"].apply(
                lambda name: _prefix_weather_dir(str(name))
            )
        )
    else:
        raise KeyError(
            "Could not derive EPW filename: expected one of "
            "['weather_station_epw_filepath', 'Region_Administrative', 'epw_filename'] in dataset CSV."
        )

    df = df.assign(
        idf_filename=[f"IDFsAndSchedules/{i}/in.idf" for i in range(1, len(df) + 1)]
    )
    df = df.assign(
        schedule_filename=[
            f"IDFsAndSchedules/{i}/in.schedules.csv" for i in range(1, len(df) + 1)
        ]
    )
    return df


def _resolve_output_path(output_dir: Path, output: str) -> Path:
    p = Path(output).expanduser()
    if p.is_absolute():
        return p
    return (output_dir / p).resolve()


def _parse_controls(raw: object) -> list[ControlName]:
    allowed: set[str] = {
        "unitary_hvac",
        "baseboard",
        "fanonoff",
        "waterheater",
        "pump",
        "airterminal",
        "controller_outdoorair",
    }
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, Sequence):
        items = list(raw)
    else:
        raise TypeError("processing.controls must be a string or a sequence of strings")

    out: list[ControlName] = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError("processing.controls entries must be strings")
        if item not in allowed:
            raise ValueError(
                f"Unknown control {item!r}. Allowed: {sorted(allowed)}"
            )
        out.append(item)  # type: ignore[arg-type]
    return out


def _to_plain_dict(cfg: object) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    try:
        from omegaconf import OmegaConf  # type: ignore

        out = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(out, dict):
            raise TypeError("Hydra config must convert to a dict at the root")
        return out  # type: ignore[return-value]
    except Exception:
        raise TypeError(
            "cfg must be a dict-like or an OmegaConf/Hydra config object"
        )

