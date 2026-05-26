#!/usr/bin/env python3
"""Plot dynamics adaptation results (Paper Figure 5).

Compares specialist, baseline, and parameterized approaches on
test buildings at each difficulty level.

Usage::

    python -m baselines.plotting.plot_dynamics_adaptation \
        --specialist-csv results_dynamics_specialist.csv \
        --baseline-csv results_dynamics_baseline.csv \
        --parameterized-csv results_dynamics_parameterized.csv \
        --output figures/fig5_dynamics_adaptation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines.plotting.common import (
    APPROACH_COLORS,
    APPROACH_LABELS,
    FIGSIZE_DOUBLE,
    apply_paper_style,
    save_figure,
)


def _load_dynamics_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist-csv", type=str, required=True)
    parser.add_argument("--baseline-csv", type=str, required=True)
    parser.add_argument("--parameterized-csv", type=str, required=True)
    parser.add_argument(
        "--output", type=str, default="figures/fig5_dynamics_adaptation"
    )
    args = parser.parse_args()

    apply_paper_style()

    approaches = {
        "specialist": _load_dynamics_csv(Path(args.specialist_csv)),
        "baseline": _load_dynamics_csv(Path(args.baseline_csv)),
        "parameterized": _load_dynamics_csv(Path(args.parameterized_csv)),
    }

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE, sharey=True)

    # Left panel: reward distribution as box plots
    ax_reward = axes[0]
    data_for_box = []
    box_labels = []
    box_colors = []
    for approach_name, rows in approaches.items():
        rewards = [float(r["reward"]) for r in rows if r.get("reward")]
        if rewards:
            data_for_box.append(rewards)
            box_labels.append(APPROACH_LABELS.get(approach_name, approach_name))
            box_colors.append(APPROACH_COLORS.get(approach_name, "gray"))

    bp = ax_reward.boxplot(
        data_for_box,
        labels=box_labels,
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_reward.set_ylabel("Episode Return")
    ax_reward.set_title("Reward Distribution")

    # Right panel: normalized score
    ax_ns = axes[1]
    for i, (approach_name, rows) in enumerate(approaches.items()):
        scores = [
            float(r["normalized_score"]) for r in rows if r.get("normalized_score")
        ]
        if scores:
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            color = APPROACH_COLORS.get(approach_name, f"C{i}")
            label = APPROACH_LABELS.get(approach_name, approach_name)
            ax_ns.bar(
                i,
                mean_s,
                yerr=std_s,
                color=color,
                label=label,
                capsize=3,
                edgecolor="white",
                linewidth=0.5,
                width=0.5,
            )

    ax_ns.set_xticks(range(len(approaches)))
    ax_ns.set_xticklabels(
        [APPROACH_LABELS.get(a, a) for a in approaches],
        rotation=15,
        ha="right",
    )
    ax_ns.set_ylabel("Normalized Score")
    ax_ns.set_title("Generalization Score")

    fig.suptitle("Dynamics Adaptation (Section 6.1)")
    fig.tight_layout()
    save_figure(fig, Path(args.output))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
