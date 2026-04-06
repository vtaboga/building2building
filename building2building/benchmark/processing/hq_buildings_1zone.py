"""
Backward-compatible wrapper for Hydro-Québec processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from building2building.benchmark.processing.hq_buildings import (  # noqa: WPS433
    BuildingProcessingRecord,
    HQBuildingsProcessingConfig,
    run_hq_buildings_processing,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deprecated CLI. Prefer `python scripts/process_hq_buildings.py` (Hydra)."
        )
    )
    p.add_argument("--dataset-zip", type=str, default="big", choices=["big", "small"])
    p.add_argument("--output", type=str, default="hq_buildings_processing_results.json")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument(
        "--controls",
        type=str,
        nargs="*",
        default=["unitary_hvac", "baseboard", "fanonoff"],
    )
    p.add_argument("--store-path", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = {
        "processing": {
            "dataset_zip": args.dataset_zip,
            "output": args.output,
            "start": args.start,
            "stop": args.stop,
            "limit": args.limit,
            "checkpoint_every": args.checkpoint_every,
            "controls": args.controls,
            "store_path": args.store_path,
            "dry_run": bool(args.dry_run),
        }
    }
    _ = run_hq_buildings_processing(cfg, output_dir=Path.cwd())


if __name__ == "__main__":
    main()