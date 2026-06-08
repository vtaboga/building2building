"""Shared plotting utilities for all paper figures.

Provides consistent styling, color palettes, and common data loading
functions so that per-figure scripts stay minimal.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

FIGSIZE_SINGLE = (4.0, 3.0)
FIGSIZE_DOUBLE = (8.0, 3.0)
FIGSIZE_WIDE = (10.0, 4.0)

BUILDING_TYPE_LABELS: dict[str, str] = {
    "OfficeSmall": "Office (S)",
    "OfficeMedium": "Office (M)",
    "RetailStandalone": "Retail",
    "RestaurantFastFood": "Restaurant",
    "Warehouse": "Warehouse",
    "SingleFamilyHouse": "House",
}

BUILDING_TYPE_ORDER = [
    "OfficeSmall",
    "OfficeMedium",
    "RetailStandalone",
    "RestaurantFastFood",
    "Warehouse",
    "SingleFamilyHouse",
]

TASK_LABELS: dict[str, str] = {
    "task1": "Task 1",
    "task2": "Task 2",
    "task3": "Task 3",
    "task4": "Task 4",
}

APPROACH_COLORS: dict[str, str] = {
    "reactive_control": "#4C72B0",
    "ppo": "#DD8452",
    "specialist": "#55A868",
    "baseline": "#C44E52",
    "parameterized": "#8172B3",
    "amorpheus": "#937860",
}

APPROACH_LABELS: dict[str, str] = {
    "reactive_control": "Reactive",
    "ppo": "PPO Specialist",
    "specialist": "Per-building",
    "baseline": "Multi-building",
    "parameterized": "Parameterized",
    "amorpheus": "Amorpheus",
}


def apply_paper_style() -> None:
    """Apply a clean paper-quality matplotlib style."""
    mpl.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class ResultRow:
    building_type: str
    task: str
    building_id: str
    reward_mean: float
    rewards: list[float]


def load_results_csv(path: Path) -> list[ResultRow]:
    """Load a baseline_returns.csv or similar results CSV."""
    rows: list[ResultRow] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            reward_cols = [k for k in raw.keys() if k.startswith("reward_run")]
            rewards = [float(raw[k]) for k in sorted(reward_cols) if raw[k]]
            rows.append(
                ResultRow(
                    building_type=raw.get("building_type", ""),
                    task=raw.get("task", ""),
                    building_id=raw.get("building_id", ""),
                    reward_mean=float(raw.get("reward_mean", 0.0)),
                    rewards=rewards,
                )
            )
    return rows


def group_by_building_type(
    rows: list[ResultRow],
) -> dict[str, list[ResultRow]]:
    """Group result rows by building type."""
    groups: dict[str, list[ResultRow]] = {}
    for r in rows:
        groups.setdefault(r.building_type, []).append(r)
    return groups


def group_by_task(rows: list[ResultRow]) -> dict[str, list[ResultRow]]:
    """Group result rows by task."""
    groups: dict[str, list[ResultRow]] = {}
    for r in rows:
        groups.setdefault(r.task, []).append(r)
    return groups


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def bar_chart(
    ax: plt.Axes,
    labels: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float] | None = None,
    color: str = "#4C72B0",
    label: str | None = None,
    offset: float = 0.0,
    width: float = 0.35,
) -> None:
    """Draw a bar chart on the given axes."""
    x = np.arange(len(labels))
    ax.bar(
        x + offset,
        values,
        width,
        yerr=errors,
        color=color,
        label=label,
        capsize=2,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")


def grouped_bar_chart(
    ax: plt.Axes,
    categories: Sequence[str],
    series: dict[str, tuple[Sequence[float], Sequence[float] | None]],
    colors: dict[str, str] | None = None,
) -> None:
    """Draw a grouped bar chart with multiple series."""
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(len(categories))

    if colors is None:
        colors = {}

    for i, (name, (vals, errs)) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * width
        color = colors.get(name, APPROACH_COLORS.get(name, f"C{i}"))
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            label=APPROACH_LABELS.get(name, name),
            color=color,
            capsize=2,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.legend()


def save_figure(
    fig: plt.Figure, path: Path, formats: Sequence[str] = ("pdf", "png")
) -> None:
    """Save figure in multiple formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out)
    plt.close(fig)
