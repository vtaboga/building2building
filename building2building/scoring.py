"""Normalized scoring relative to baseline controller performance.

Implements the scoring methodology from Section 5 of the Building2Building
paper: cumulative episode return divided by the reactive-controller baseline
return for the same building, task, and building type.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

BuildingType = Literal[
    "SingleFamilyHouse",
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

_baseline_cache: dict[tuple[str, str, int], float] | None = None
_baseline_cache_no_task: dict[tuple[str, int], float] | None = None


def _load_baseline_returns() -> (
    tuple[dict[tuple[str, str, int], float], dict[tuple[str, int], float]]
):
    """Load baseline returns from the package-shipped CSV.

    Supports two CSV formats:
    - **With task column:** ``building_type,task,building_id,...,reward_mean``
    - **Legacy (no task):** ``building_type,building_id,...,reward_mean``

    Returns a tuple of (task-keyed dict, no-task-keyed dict).  When the CSV
    has no ``task`` column the task-keyed dict is empty.
    """
    global _baseline_cache, _baseline_cache_no_task
    if _baseline_cache is not None and _baseline_cache_no_task is not None:
        return _baseline_cache, _baseline_cache_no_task

    csv_path = Path(__file__).parent.parent / "baseline_returns.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Baseline returns file not found at {csv_path}. "
            "This file should ship with the repository."
        )

    with_task: dict[tuple[str, str, int], float] = {}
    no_task: dict[tuple[str, int], float] = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        has_task_col = "task" in (reader.fieldnames or [])

        for row in reader:
            bt = row["building_type"]
            bid = int(row["building_id"])
            mean_return = float(row["reward_mean"])

            if has_task_col and row.get("task"):
                task_name = row["task"]
                with_task[(bt, task_name, bid)] = mean_return
            no_task[(bt, bid)] = mean_return

    _baseline_cache = with_task
    _baseline_cache_no_task = no_task
    return with_task, no_task


def compute_normalized_score(
    cumulative_return: float,
    building_type: BuildingType,
    task: str,
    building_id: int | None = None,
) -> float:
    """Compute a normalized score relative to the reactive-controller baseline.

    The score is ``cumulative_return / baseline_return``, where
    ``baseline_return`` is the mean return of the reactive controller
    on the same building (or averaged over the building type).

    A score of 1.0 means the agent matches the baseline; higher is better
    (less negative return = better performance for cost-based rewards).

    When the CSV includes per-task baselines, the *task* argument is used
    to select the matching baseline.  Otherwise it falls back to task-
    agnostic baselines for backward compatibility.

    Args:
        cumulative_return: Total episode return achieved by the agent.
        building_type: The building type used.
        task: The task preset name (e.g. ``"task1"``).
        building_id: Optional specific building ID for per-building
            normalization.  If ``None``, uses the average baseline
            return for the building type.

    Returns:
        The normalized score (dimensionless ratio).

    Raises:
        KeyError: If no baseline data exists for the given building.
    """
    with_task, no_task = _load_baseline_returns()

    baseline: float | None = None

    if building_id is not None:
        task_key = (building_type, task, building_id)
        if task_key in with_task:
            baseline = with_task[task_key]
        else:
            notask_key = (building_type, building_id)
            if notask_key in no_task:
                baseline = no_task[notask_key]
            else:
                raise KeyError(
                    f"No baseline return found for building_type={building_type!r}, "
                    f"task={task!r}, building_id={building_id}"
                )
    else:
        task_baselines = [
            v for (bt, t, _), v in with_task.items()
            if bt == building_type and t == task
        ]
        if task_baselines:
            baseline = sum(task_baselines) / len(task_baselines)
        else:
            type_baselines = [
                v for (bt, _), v in no_task.items() if bt == building_type
            ]
            if not type_baselines:
                raise KeyError(
                    f"No baseline returns found for building_type={building_type!r}"
                )
            baseline = sum(type_baselines) / len(type_baselines)

    if baseline == 0.0:
        logger.warning(
            "Baseline return is 0 for %s/%s; returning raw cumulative_return.",
            building_type,
            task,
        )
        return cumulative_return

    return cumulative_return / baseline
