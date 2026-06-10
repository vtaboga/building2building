#!/usr/bin/env python3
"""Concatenate per-job baseline-return CSVs into the master file.

Intended to run after ``sbatch baselines/scripts/run_baseline_returns.sh``
has produced one CSV per ``(building_type, task)`` under ``$SCRATCH``.

Usage::

    python baselines/scripts/merge_baseline_returns.py "$SCRATCH/b2b/baseline_returns"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DST = PROJECT_ROOT / "building2building" / "scores" / "baseline_returns.csv"

N_BUILDING_TYPES = 6
N_TASKS = 9  # 3x3 grid: (const, occ, rand) x (e0, emed, ehigh)
N_RUN_PERIODS = 3
N_BUILDINGS_PER_TYPE = 100
EXPECTED_ROWS = N_BUILDING_TYPES * N_TASKS * N_RUN_PERIODS * N_BUILDINGS_PER_TYPE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src",
        type=Path,
        help="Directory containing per-job CSVs (e.g. $SCRATCH/b2b/baseline_returns).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Output CSV path (default: {DEFAULT_DST}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Assert that exactly %d rows were merged." % EXPECTED_ROWS,
    )
    args = parser.parse_args()

    csvs = sorted(args.src.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV files found under {args.src}")

    df = pd.concat((pd.read_csv(p) for p in csvs), ignore_index=True)

    df.sort_values(
        ["building_type", "task", "run_period", "building_id"],
        inplace=True,
        kind="mergesort",
    )
    df.drop_duplicates(
        subset=["building_type", "task", "run_period", "building_id"],
        keep="last",
        inplace=True,
    )

    print(f"Merged {len(csvs)} files -> {len(df)} rows")
    if args.strict and len(df) != EXPECTED_ROWS:
        raise SystemExit(
            f"Expected {EXPECTED_ROWS} rows, got {len(df)}. "
            "Re-run failed jobs or drop --strict."
        )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dst, index=False)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
