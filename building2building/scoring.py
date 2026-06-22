"""Normalized scoring relative to baseline reactive controller performance."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Literal

from building2building.data.download import BuildingType

logger = logging.getLogger(__name__)

RunPeriod = Literal["full_year", "winter", "summer"]

CSV_PATH = Path(__file__).parent / "scores" / "baseline_returns.csv"

_cache: dict[tuple[str, str, str, str], float] | None = None


def _load() -> dict[tuple[str, str, str, str], float]:
    """Load baseline returns keyed by ``(building_type, task, run_period, building_id)``."""
    global _cache
    if _cache is not None:
        return _cache

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Baseline returns file not found at {CSV_PATH}. "
            "Run `sbatch baselines/scripts/run_baseline_returns.sh` then "
            "`python baselines/scripts/merge_baseline_returns.py $SCRATCH/b2b_baseline_returns`."
        )

    out: dict[tuple[str, str, str, str], float] = {}
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "building_type",
            "task",
            "run_period",
            "building_id",
            "reward_mean",
        }
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            raise KeyError(
                f"Baseline CSV at {CSV_PATH} is missing required columns: {missing_columns}"
            )
        for row in reader:
            key = (
                row["building_type"],
                row["task"],
                row["run_period"],
                str(row["building_id"]),
            )
            out[key] = float(row["reward_mean"])

    _cache = out
    return out


def compute_normalized_score(
    cumulative_return: float,
    building_type: BuildingType,
    task: str,
    run_period: RunPeriod,
    building_id: str,
) -> float:
    """Compute a normalized score relative to the reactive-controller baseline.

    The score is ``cumulative_return / baseline_return``, where
    ``baseline_return`` is the reactive controller's return on the same
    ``(building_type, task, run_period, building_id)``.

    A score of 1.0 means the agent matches the baseline; **lower is
    better**.  Because both returns are negative (cost-based rewards), a
    less-negative agent return yields a smaller ratio: a score < 1.0 beats
    the reactive baseline and > 1.0 is worse than it.

    Args:
        cumulative_return: Total episode return achieved by the agent.
        building_type: The building type used.
        task: The task preset name (e.g. ``"task_const_e0"``).
        run_period: Simulation run period. One of ``"full_year"``,
            ``"winter"``, ``"summer"``.
        building_id: Specific building ID for per-building normalization.

    Returns:
        The normalized score (dimensionless ratio).

    Raises:
        ValueError: If ``building_id`` is ``None``.
        KeyError: If no baseline data exists for the requested key.
    """
    if building_id is None:
        raise ValueError("building_id must be specified to compute a normalized score.")

    baselines = _load()

    key = (building_type, task, run_period, building_id)
    if key not in baselines:
        raise KeyError(
            f"No baseline return found for building_type={building_type!r}, "
            f"task={task!r}, run_period={run_period!r}, building_id={building_id!r}"
        )
    baseline = baselines[key]

    if baseline == 0.0:
        logger.warning(
            "Baseline return is 0 for %s/%s/%s; returning raw cumulative_return.",
            building_type,
            task,
            run_period,
        )
        return cumulative_return

    return cumulative_return / baseline
