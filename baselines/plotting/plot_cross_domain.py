#!/usr/bin/env python3
"""Plot cross-domain transfer results (Paper Figure 7).

Shows zero-shot transfer performance across building types.

Usage::

    python -m baselines.plotting.plot_cross_domain \
        --results-csv results_cross_domain.csv \
        --output figures/fig7_cross_domain
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines.plotting.common import (
    BUILDING_TYPE_LABELS,
    FIGSIZE_SINGLE,
    apply_paper_style,
    save_figure,
    APPROACH_COLORS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/fig7_cross_domain")
    args = parser.parse_args()

    apply_paper_style()

    rows: list[dict[str, str]] = []
    with Path(args.results_csv).open() as f:
        rows = list(csv.DictReader(f))

    by_type: dict[str, list[float]] = {}
    for r in rows:
        bt = r["building_type"]
        by_type.setdefault(bt, []).append(float(r["reward"]))

    types = list(by_type.keys())
    labels = [BUILDING_TYPE_LABELS.get(bt, bt) for bt in types]
    means = [np.mean(by_type[bt]) for bt in types]
    stds = [np.std(by_type[bt]) for bt in types]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    x = np.arange(len(types))
    ax.bar(
        x,
        means,
        yerr=stds,
        color=APPROACH_COLORS["amorpheus"],
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
        width=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Episode Return")
    ax.set_title("Cross-Domain Transfer (Amorpheus)")
    fig.tight_layout()

    save_figure(fig, Path(args.output))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
