#!/usr/bin/env python3
"""Gather trained PPO models from scratch into a canonical directory layout.

Iterates over all 12 tag × 8 building combinations, copies the best available
model and its Hydra config into::

    <dest>/
      <BuildingType>/
        <task>/
          building_<idx>/
            model.zip
            config.yaml

Usage:
    python scripts/gather_trained_models.py
    python scripts/gather_trained_models.py --dest /path/to/trained_models
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TAG_TO_TASK: dict[str, str] = {
    "wh_db001_const": "deadband_ew001_const",
    "wh_db001_occ": "deadband_ew001_occ",
    "wh_db01_const": "deadband_ew01_const",
    "wh_bar": "barrier_ew1",
    "rt_db001_const": "deadband_ew001_const",
    "rt_db001_occ": "deadband_ew001_occ",
    "rt_db01_const": "deadband_ew01_const",
    "rt_bar": "barrier_ew1",
    "rf_db001_const": "deadband_ew001_const",
    "rf_db001_occ": "deadband_ew001_occ",
    "rf_db01_const": "deadband_ew01_const",
    "rf_bar": "barrier_ew1",
}

TAG_TO_BUILDING_TYPE: dict[str, str] = {
    "wh_db001_const": "Warehouse",
    "wh_db001_occ": "Warehouse",
    "wh_db01_const": "Warehouse",
    "wh_bar": "Warehouse",
    "rt_db001_const": "RetailStandalone",
    "rt_db001_occ": "RetailStandalone",
    "rt_db01_const": "RetailStandalone",
    "rt_bar": "RetailStandalone",
    "rf_db001_const": "RestaurantFastFood",
    "rf_db001_occ": "RestaurantFastFood",
    "rf_db01_const": "RestaurantFastFood",
    "rf_bar": "RestaurantFastFood",
}

NUM_BUILDINGS = 8
SCRATCH_BASE = Path(
    "/network/scratch/v/vincent.taboga/Building2Building/outputs"
)
DEFAULT_DEST = Path(
    "/network/scratch/v/vincent.taboga/Building2Building/trained_models"
)


StatusKind = Literal["best_model", "final_model", "checkpoint", "missing"]


@dataclass
class GatherResult:
    tag: str
    building_index: int
    building_type: str
    task: str
    status: StatusKind
    source_model: Path | None
    dest_dir: Path | None


def _pick_model_from_run(run_dir: Path) -> tuple[StatusKind, Path | None]:
    """Return the best available model from a training run directory."""
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        return "missing", None

    for name, kind in [
        ("best_model.zip", "best_model"),
        ("final_model.zip", "final_model"),
        ("model.zip", "best_model"),
    ]:
        p = models_dir / name
        if p.exists():
            return kind, p  # type: ignore[return-value]

    checkpoints = sorted(models_dir.glob("checkpoint_*_steps.zip"))
    if checkpoints:
        return "checkpoint", checkpoints[-1]

    return "missing", None


def gather_models(
    scratch_base: Path,
    dest: Path,
) -> list[GatherResult]:
    results: list[GatherResult] = []

    for tag in TAG_TO_TASK:
        building_type = TAG_TO_BUILDING_TYPE[tag]
        task = TAG_TO_TASK[tag]

        for idx in range(NUM_BUILDINGS):
            run_dir = scratch_base / tag / f"building_{idx}"
            status, model_path = _pick_model_from_run(run_dir)
            cfg_path = run_dir / ".hydra" / "config.yaml"

            if status == "missing" or model_path is None:
                results.append(
                    GatherResult(
                        tag=tag,
                        building_index=idx,
                        building_type=building_type,
                        task=task,
                        status="missing",
                        source_model=None,
                        dest_dir=None,
                    )
                )
                continue

            out_dir = dest / building_type / task / f"building_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(model_path, out_dir / "model.zip")
            if cfg_path.exists():
                shutil.copy2(cfg_path, out_dir / "config.yaml")

            results.append(
                GatherResult(
                    tag=tag,
                    building_index=idx,
                    building_type=building_type,
                    task=task,
                    status=status,
                    source_model=model_path,
                    dest_dir=out_dir,
                )
            )

    return results


def print_summary(results: list[GatherResult]) -> None:
    total = len(results)
    found = sum(1 for r in results if r.status != "missing")
    missing = total - found

    print(f"\n{'=' * 70}")
    print(f"Gather summary: {found}/{total} models copied, {missing} missing")
    print(f"{'=' * 70}\n")

    header = f"{'Tag':<20} {'Idx':>3} {'Type':<20} {'Task':<25} {'Status':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        marker = "OK" if r.status != "missing" else "MISSING"
        src = ""
        if r.source_model is not None:
            src = f"  ({r.source_model.name})"
        print(
            f"{r.tag:<20} {r.building_index:>3} {r.building_type:<20} "
            f"{r.task:<25} {marker:<12}{src}"
        )

    if missing > 0:
        print(f"\nMissing models ({missing}):")
        for r in results:
            if r.status == "missing":
                print(f"  {r.tag}/building_{r.building_index}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=SCRATCH_BASE,
        help="Root directory of training outputs on scratch.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination root for gathered models.",
    )
    args = parser.parse_args()

    results = gather_models(args.scratch, args.dest)
    print_summary(results)


if __name__ == "__main__":
    main()
