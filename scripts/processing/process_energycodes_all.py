"""
Process every active ASHRAE 90.1 building through the pipeline and record
actuator metadata.

For each building we store:
- building_type, year, place, filename (identification)
- actuator descriptions found by make_controllable
- success flag and error message on failure

Outputs are written to ``outputs/energycodes_processing_results.json``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args

from building2building.env import STORE_PATH
from building2building.sources.energycodes import BuildingType, search_buildings
from building2building.store import realize

ACTIVE_BUILDING_TYPES: list[str] = list(get_args(BuildingType))


@dataclass(frozen=True)
class ProcessingRecord:
    index: int
    building_type: str
    year: int
    place: str
    filename: str
    actuators: list[dict[str, Any]]
    success: bool
    error: str | None


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


def _extract_actuators(equipment: Any) -> list[dict[str, Any]]:
    """Pull actuator dicts from whatever make_controllable returned."""
    out: list[dict[str, Any]] = []
    for item in equipment:
        if hasattr(item, "actuator_descriptions"):
            for a in item.actuator_descriptions():
                out.append({
                    "component_type": a.component_type,
                    "control_type": a.control_type,
                    "component_name": a.component_name,
                    "units": a.units,
                    "lower_bound": a.lower_bound,
                    "upper_bound": a.upper_bound,
                })
        elif hasattr(item, "component_type"):
            out.append({
                "component_type": item.component_type,
                "control_type": item.control_type,
                "component_name": item.component_name,
                "units": item.units,
                "lower_bound": item.lower_bound,
                "upper_bound": item.upper_bound,
            })
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process every ASHRAE 90.1 building and record actuator metadata."
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/energycodes_processing_results.json",
        help="Where to write the JSON summary (relative to cwd).",
    )
    p.add_argument(
        "--building-type",
        type=str,
        default=None,
        help="Process only this building type (e.g. OfficeSmall).",
    )
    p.add_argument(
        "--one-per-type",
        action="store_true",
        help="Process only the first building of each active type.",
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
    output_path = Path(args.output).resolve()

    all_rows = search_buildings()
    active = all_rows[all_rows["building_type"].isin(ACTIVE_BUILDING_TYPES)]

    if args.building_type is not None:
        if args.building_type not in ACTIVE_BUILDING_TYPES:
            raise ValueError(
                f"{args.building_type!r} is not an active building type. "
                f"Active: {ACTIVE_BUILDING_TYPES}"
            )
        active = active[active["building_type"] == args.building_type]

    if args.one_per_type:
        active = active.groupby("building_type").first().reset_index()
    else:
        active = active.reset_index(drop=True)
    total = len(active)
    print(f"Processing {total} buildings across {active['building_type'].nunique()} types")

    records: list[ProcessingRecord] = []

    for _, row in active.iterrows():
        i = len(records) + 1
        label = f"{row.building_type} / {row.year} / {row.place}"
        print(f"[{i}/{total}] {label}")

        try:
            epjson_path, equipment = realize(STORE_PATH.get(), row.derivation_thunk())
            actuators = _extract_actuators(equipment)

            rec = ProcessingRecord(
                index=int(row["index"]),
                building_type=row.building_type,
                year=int(row.year),
                place=row.place,
                filename=row.filename,
                actuators=actuators,
                success=True,
                error=None,
            )
        except Exception as e:
            traceback.print_exc()
            rec = ProcessingRecord(
                index=int(row["index"]),
                building_type=row.building_type,
                year=int(row.year),
                place=row.place,
                filename=row.filename,
                actuators=[],
                success=False,
                error=_short_error(e),
            )

        records.append(rec)

        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            _atomic_write_json(output_path, [asdict(r) for r in records])

    _atomic_write_json(output_path, [asdict(r) for r in records])

    n_ok = sum(1 for r in records if r.success)
    n_fail = len(records) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed out of {len(records)}.")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
