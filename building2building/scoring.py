"""Normalized scoring relative to baseline controller performance.

Implements the scoring methodology from Section 5 of the Building2Building
paper: cumulative episode return divided by the reactive-controller baseline
return for the same building and task.
"""

from __future__ import annotations

import csv
import logging
from importlib.resources import files
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

_baseline_cache: dict[tuple[str, int], float] | None = None


def _load_baseline_returns() -> dict[tuple[str, int], float]:
    """Load baseline returns from the package-shipped CSV."""
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache

    csv_path = Path(__file__).parent.parent / "baseline_returns.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Baseline returns file not found at {csv_path}. "
            "This file should ship with the repository."
        )

    result: dict[tuple[str, int], float] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bt = row["building_type"]
            bid = int(row["building_id"])
            mean_return = float(row["reward_mean"])
            result[(bt, bid)] = mean_return

    _baseline_cache = result
    return result


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
    baselines = _load_baseline_returns()

    if building_id is not None:
        key = (building_type, building_id)
        if key not in baselines:
            raise KeyError(
                f"No baseline return found for building_type={building_type!r}, "
                f"building_id={building_id}"
            )
        baseline = baselines[key]
    else:
        type_baselines = [v for (bt, _), v in baselines.items() if bt == building_type]
        if not type_baselines:
            raise KeyError(
                f"No baseline returns found for building_type={building_type!r}"
            )
        baseline = sum(type_baselines) / len(type_baselines)

    if baseline == 0.0:
        logger.warning(
            "Baseline return is 0 for %s; returning raw cumulative_return.",
            building_type,
        )
        return cumulative_return

    return cumulative_return / baseline
