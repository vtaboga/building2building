#!/usr/bin/env python3
"""Plot per-building-type PPO specialist results (Paper Figure 4).

Reads the PPO evaluation CSV and the reactive control baseline CSV, then
produces a grouped bar chart comparing normalized scores across
building types and tasks.

Usage::

    python -m baselines.plotting.plot_ppo_specialist \
        --ppo-csv results_ppo.csv \
        --baseline-csv baseline_returns.csv \
        --output figures/fig4_ppo_specialist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines.plotting.common import (
    BUILDING_TYPE_LABELS,
    BUILDING_TYPE_ORDER,
    apply_paper_style,
    grouped_bar_chart,
    load_results_csv,
    save_figure,
    FIGSIZE_WIDE,
    group_by_building_type,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-csv", type=str, required=True)
    parser.add_argument("--baseline-csv", type=str, required=True)
    parser.add_argument("--task", type=str, default="task1")
    parser.add_argument("--output", type=str, default="figures/fig4_ppo_specialist")
    args = parser.parse_args()

    apply_paper_style()

    ppo_rows = load_results_csv(Path(args.ppo_csv))
    baseline_rows = load_results_csv(Path(args.baseline_csv))

    ppo_by_type = group_by_building_type([r for r in ppo_rows if r.task == args.task])
    baseline_by_type = group_by_building_type(
        [r for r in baseline_rows if r.task == args.task]
    )

    types_present = [bt for bt in BUILDING_TYPE_ORDER if bt in ppo_by_type]
    labels = [BUILDING_TYPE_LABELS.get(bt, bt) for bt in types_present]

    baseline_means = []
    baseline_stds = []
    ppo_means = []
    ppo_stds = []

    for bt in types_present:
        bl = baseline_by_type.get(bt, [])
        bl_rewards = [r.reward_mean for r in bl]
        baseline_means.append(np.mean(bl_rewards) if bl_rewards else 0)
        baseline_stds.append(np.std(bl_rewards) if bl_rewards else 0)

        pp = ppo_by_type.get(bt, [])
        pp_rewards = [r.reward_mean for r in pp]
        ppo_means.append(np.mean(pp_rewards) if pp_rewards else 0)
        ppo_stds.append(np.std(pp_rewards) if pp_rewards else 0)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    grouped_bar_chart(
        ax,
        labels,
        {
            "reactive_control": (baseline_means, baseline_stds),
            "ppo": (ppo_means, ppo_stds),
        },
    )
    ax.set_ylabel("Episode Return")
    ax.set_title(f"PPO Specialist vs Reactive ({args.task})")

    save_figure(fig, Path(args.output))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
