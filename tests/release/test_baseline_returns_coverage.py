from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from building2building import scoring
from building2building.data.registry import get_registry


def _load_eval_grid() -> tuple[list[str], list[str], list[str], str]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root / "baselines" / "configs" / "experiment" / "eval_reactive_control.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return (
        list(cfg["building_types"]),
        list(cfg["tasks"]),
        list(cfg["run_periods"]),
        str(cfg["split"]),
    )


@pytest.mark.release
def test_baseline_returns_cover_paper_evaluation_grid() -> None:
    scoring._cache = None
    baselines = scoring._load()
    registry = get_registry()
    building_types, tasks, run_periods, split = _load_eval_grid()
    assert split == "test"

    expected: set[tuple[str, str, str, str]] = set()
    for building_type in building_types:
        building_ids = registry.list_buildings(building_type, "test")
        for task in tasks:
            for run_period in run_periods:
                for building_id in building_ids:
                    expected.add((building_type, task, run_period, building_id))

    missing = sorted(expected - set(baselines.keys()))
    assert missing == [], f"Missing baseline rows for expected tuples: {missing[:10]}"
