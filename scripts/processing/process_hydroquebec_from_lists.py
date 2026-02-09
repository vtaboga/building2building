"""
Process a subset of Hydro-Québec buildings defined by stored selection lists.

These lists live in `b2b/sources/data/` and are loaded through
`b2b.utils.HydroQuebecRowIdSplits`.

For each processed building, we store:
- building id
- (derived) dataset_row_index
- number of conditioned zones and unconditioned zones
- actuator component_name strings
- success flag, and a short error message on failure

Outputs are written to a JSON file (typically under `outputs/` or $SCRATCH).

IMPORTANT: This script forces caching into $SCRATCH (not $HOME).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class BuildingProcessingRecord:
    split: str
    split_index: int
    building_id: int
    dataset_row_index: int
    n_conditioned_zones: int
    n_unconditioned_zones: int
    actuator_names: list[str]
    success: bool
    error: str | None


def _require_scratch_store() -> Path:
    scratch_raw = os.environ.get("SCRATCH")
    if not scratch_raw:
        raise RuntimeError(
            "SCRATCH is not set. Refusing to run because this script must cache "
            "processed files in $SCRATCH (not in $HOME)."
        )

    scratch = Path(scratch_raw).expanduser().resolve()
    store_path_raw = os.environ.get("STORE_PATH")
    if store_path_raw:
        store_path = Path(store_path_raw).expanduser().resolve()
    else:
        store_path = (scratch / "Building2Building" / "store").resolve()

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
        set(epjson.get("Zone", {}).keys()) if isinstance(epjson.get("Zone"), dict) else set()
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process Hydro-Québec buildings from stored selection lists."
    )
    p.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Which stored list to use (train/test/all=concatenate train+test).",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index within the selected list (0-based).",
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop index within the selected list (0-based, exclusive).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/hydroquebec_processing_results_from_lists.json",
        help="Where to write the JSON output.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Rewrite the JSON output every N buildings.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # scripts/processing/<this_file>.py -> repo root is two levels up
    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    store_path = _require_scratch_store()

    # Import after forcing STORE_PATH.
    from b2b.env import STORE_PATH  # noqa: WPS433
    from b2b.env import energyplus_path  # noqa: WPS433
    from b2b.sources import hydroquebec  # noqa: WPS433
    from b2b.store import realize  # noqa: WPS433
    from b2b.utils import (  # noqa: WPS433
        HydroQuebecRowIdSplits,
    )

    STORE_PATH.set(store_path)

    splits = HydroQuebecRowIdSplits.load_from_action_space_2_zone_1()
    split: Literal["train", "test", "all"] = args.split
    if split == "train":
        row_indices: list[int] = list(splits.train_row_ids)
        split_tags: list[str] = ["train"] * len(row_indices)
        split_indices: list[int] = list(range(len(row_indices)))
    elif split == "test":
        row_indices = list(splits.test_row_ids)
        split_tags = ["test"] * len(row_indices)
        split_indices = list(range(len(row_indices)))
    else:
        row_indices = list(splits.train_row_ids) + list(splits.test_row_ids)
        split_tags = (["train"] * len(splits.train_row_ids)) + (
            ["test"] * len(splits.test_row_ids)
        )
        split_indices = list(range(len(row_indices)))

    row_indices_sliced = row_indices[args.start : args.stop]
    tags_sliced = split_tags[args.start : args.stop]
    split_idx_sliced = split_indices[args.start : args.stop]

    ep = energyplus_path()
    root_zip = hydroquebec.dataset_zip()

    records: list[BuildingProcessingRecord] = []
    processed = 0

    for tag, split_index, dataset_row_index in zip(
        tags_sliced, split_idx_sliced, row_indices_sliced
    ):
        building_id = int(dataset_row_index) + 1
        try:
            idf_filename = f"IDFsAndSchedules/{building_id}/in.idf"
            schedule_filename = f"IDFsAndSchedules/{building_id}/in.schedules.csv"
            deriv = hydroquebec._build_control_derivation(  # noqa: SLF001
                root_zip, idf_filename, schedule_filename, ep
            )
            epjson_path, actuator_descs = realize(STORE_PATH.get(), deriv)

            epjson = _load_epjson(Path(epjson_path))
            n_cond, n_uncond = _get_conditioned_and_unconditioned_zone_counts(epjson)

            actuator_names: list[str] = []
            for a in actuator_descs:
                try:
                    actuator_names.append(str(a.component_name))
                except Exception:
                    actuator_names.append(str(a))

            rec = BuildingProcessingRecord(
                split=tag,
                split_index=int(split_index),
                building_id=building_id,
                dataset_row_index=int(dataset_row_index),
                n_conditioned_zones=n_cond,
                n_unconditioned_zones=n_uncond,
                actuator_names=actuator_names,
                success=True,
                error=None,
            )
        except Exception as e:
            rec = BuildingProcessingRecord(
                split=tag,
                split_index=int(split_index),
                building_id=building_id,
                dataset_row_index=int(dataset_row_index),
                n_conditioned_zones=0,
                n_unconditioned_zones=0,
                actuator_names=[],
                success=False,
                error=_short_error(e),
            )

        records.append(rec)
        processed += 1
        if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
            _atomic_write_json(out_path, [r.__dict__ for r in records])

    _atomic_write_json(out_path, [r.__dict__ for r in records])


if __name__ == "__main__":
    main()

