"""
Build Hydro-Québec train/test selection lists from existing processing outputs.

We filter processed buildings to those with exactly two actuators:
- exactly one fan actuator
- exactly one temperature node setpoint actuator

Then we sample non-overlapping train/test lists (default 900/100) and save them
as pickled int lists under `b2b/sources/data/`.

IMPORTANT:
The saved lists are *dataset row indices* (0-based), not building ids.
Downstream selection maps:
  building_id = dataset_row_index + 1
to resolve `IDFsAndSchedules/<building_id>/...`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Record:
    dataset_row_index: int
    actuator_names: list[str]
    success: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Hydro-Québec train/test row-index lists from JSON chunks."
    )
    p.add_argument(
        "--inputs-dir",
        type=str,
        default="b2b_outputs",
        help="Directory containing hydroquebec_processing_results_*.json chunk files.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="hydroquebec_processing_results*.json",
        help="Glob pattern within inputs-dir to match JSON chunk files.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible sampling.",
    )
    p.add_argument(
        "--train-size",
        type=int,
        default=900,
        help="Number of train indices to sample.",
    )
    p.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of test indices to sample.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="b2b/sources/data",
        help="Where to write the pickled lists (relative to repo root).",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="action_space_2_zone_1",
        help="Output prefix, writes <prefix>_train_data and <prefix>_test_data.",
    )
    p.add_argument(
        "--report-json",
        type=str,
        default="outputs/hq_selection_list_report.json",
        help="Where to write a small summary report JSON.",
    )
    return p.parse_args()


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"Expected JSON list in {path}, got {type(obj).__name__}")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected list[dict] in {path} at index={i}, got {type(item).__name__}"
            )
        out.append(item)
    return out


def _coerce_record(d: dict[str, Any], *, src: Path) -> Record | None:
    success = bool(d.get("success", False))
    if not success:
        return None

    idx = d.get("dataset_row_index")
    if not isinstance(idx, int):
        return None
    if idx < 0:
        return None

    names_any = d.get("actuator_names", [])
    if not isinstance(names_any, list):
        return None
    names = [str(x) for x in names_any]
    return Record(dataset_row_index=int(idx), actuator_names=names, success=True)


def _classify_two_actuators(actuator_names: list[str]) -> tuple[int, int, int]:
    """
    Return counts: (fans, nodes, others).

    We only have component_name strings in the processing JSON outputs, so we use
    best-effort heuristics:
    - fan: contains 'fan'
    - node temp setpoint: contains 'unitaryhvac schedule for node' OR contains both 'node' and 'setpoint'
    """
    fans = 0
    nodes = 0
    others = 0

    for a in actuator_names:
        s = str(a).strip().lower()
        if "fan" in s:
            fans += 1
        elif "unitaryhvac schedule for node" in s:
            nodes += 1
        elif "node" in s and "setpoint" in s:
            nodes += 1
        else:
            others += 1
    return fans, nodes, others


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.is_absolute():
        inputs_dir = (repo_root / inputs_dir).resolve()

    paths = sorted(inputs_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"No input JSON files matched {args.pattern!r} under {inputs_dir}"
        )

    records: dict[int, Record] = {}
    total_items = 0
    total_success = 0

    for p in paths:
        for d in _load_json_list(p):
            total_items += 1
            r = _coerce_record(d, src=p)
            if r is None:
                continue
            total_success += 1
            # Deduplicate by dataset_row_index; keep first seen.
            records.setdefault(r.dataset_row_index, r)

    eligible: list[int] = []
    for idx, r in records.items():
        if len(r.actuator_names) != 2:
            continue
        fan_c, node_c, other_c = _classify_two_actuators(r.actuator_names)
        if fan_c == 1 and node_c == 1 and other_c == 0:
            eligible.append(idx)

    eligible = sorted(set(eligible))

    need = int(args.train_size) + int(args.test_size)
    if len(eligible) < need:
        raise RuntimeError(
            f"Not enough eligible buildings: need {need}, found {len(eligible)}. "
            "Consider loosening heuristics or generating more processing outputs."
        )

    rng = random.Random(int(args.seed))
    shuffled = eligible[:]
    rng.shuffle(shuffled)
    picked = shuffled[:need]

    test_ids = sorted(picked[: int(args.test_size)])
    train_ids = sorted(picked[int(args.test_size) :])

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{args.out_prefix}_train_data"
    test_path = out_dir / f"{args.out_prefix}_test_data"
    train_path.write_bytes(pickle.dumps(train_ids))
    test_path.write_bytes(pickle.dumps(test_ids))

    report_path = Path(args.report_json)
    if not report_path.is_absolute():
        report_path = (repo_root / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "inputs_dir": str(inputs_dir),
        "pattern": str(args.pattern),
        "n_files": len(paths),
        "total_items": total_items,
        "total_success": total_success,
        "n_unique_success": len(records),
        "n_eligible_two_actuators": len(eligible),
        "seed": int(args.seed),
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
        "out_dir": str(out_dir),
        "train_file": str(train_path),
        "test_file": str(test_path),
        "note": "Stored ids are dataset row indices (0-based). building_id = row_index + 1.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote train list: {train_path} ({len(train_ids)})")
    print(f"Wrote test list:  {test_path} ({len(test_ids)})")
    print(f"Wrote report:     {report_path}")


if __name__ == "__main__":
    main()

