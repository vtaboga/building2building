#!/usr/bin/env python3
"""Visualize CHS PPO tuning results.

Reads the re-evaluation YAML summaries produced by ``tune_ppo.py``
(with ``reeval=true``) and generates:

1. Per-building bar chart of mean reward with bootstrap 95 % CIs.
2. CDF-normalized performance comparison across buildings.
3. Comparison of CHS-tuned PPO vs the reactive-controller baseline.

Usage::

    python -m baselines.plotting.plot_chs_results \
        --reeval-dir outputs/tune_ppo/.../reeval \
        --output-dir plots/chs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from baselines.chs import cdf_normalize_batch
from baselines.plotting.common import (
    APPROACH_COLORS,
    BUILDING_TYPE_LABELS,
    FIGSIZE_DOUBLE,
    FIGSIZE_SINGLE,
    apply_paper_style,
    save_figure,
)


def _load_reeval(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _bootstrap_ci(
    data: np.ndarray, n_boot: int = 10_000, ci: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    means = np.array(
        [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(
        np.percentile(means, 100 * (1 - alpha))
    )


# ── Plot 1: Per-building reward bar chart ────────────────────────────


def plot_per_building_rewards(summary: dict[str, Any], output_dir: Path) -> None:
    """Bar chart of mean reward per building with bootstrap 95 % CIs."""
    apply_paper_style()

    buildings = summary["buildings"]
    bids = sorted(buildings.keys())
    means: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []

    for bid in bids:
        rewards = np.array(buildings[bid]["rewards"])
        m = float(rewards.mean())
        means.append(m)
        if len(rewards) >= 2:
            lo, hi = _bootstrap_ci(rewards)
            ci_lo.append(m - lo)
            ci_hi.append(hi - m)
        else:
            ci_lo.append(0.0)
            ci_hi.append(0.0)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    x = np.arange(len(bids))
    ax.bar(
        x,
        means,
        yerr=[ci_lo, ci_hi],
        color=APPROACH_COLORS.get("ppo", "#DD8452"),
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    short_bids = [b[-8:] for b in bids]
    ax.set_xticks(x)
    ax.set_xticklabels(short_bids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Episode return")
    bt = summary.get("building_type", "")
    task = summary.get("task", "")
    label = BUILDING_TYPE_LABELS.get(bt, bt)
    ax.set_title(f"CHS PPO re-evaluation: {label} / {task}")
    fig.tight_layout()
    save_figure(fig, output_dir / f"reeval_rewards_{bt}_{task}")


# ── Plot 2: CDF-normalized performance ───────────────────────────────


def plot_cdf_normalized(summary: dict[str, Any], output_dir: Path) -> None:
    """CDF-normalized performance bar chart across buildings."""
    apply_paper_style()

    buildings = summary["buildings"]
    bids = sorted(buildings.keys())

    # Build a global pool of all rewards across all buildings.
    global_pool = np.array([r for b in buildings.values() for r in b["rewards"]])

    cdf_means: list[float] = []
    cdf_errors: list[float] = []
    for bid in bids:
        rewards = np.array(buildings[bid]["rewards"])
        normed = cdf_normalize_batch(rewards, global_pool)
        cdf_means.append(float(normed.mean()))
        if len(normed) >= 2:
            lo, hi = _bootstrap_ci(normed)
            cdf_errors.append((hi - lo) / 2)
        else:
            cdf_errors.append(0.0)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    x = np.arange(len(bids))
    ax.bar(
        x,
        cdf_means,
        yerr=cdf_errors,
        color=APPROACH_COLORS.get("specialist", "#55A868"),
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    short_bids = [b[-8:] for b in bids]
    ax.set_xticks(x)
    ax.set_xticklabels(short_bids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CDF-normalized score")
    ax.set_ylim(0, 1)
    bt = summary.get("building_type", "")
    task = summary.get("task", "")
    label = BUILDING_TYPE_LABELS.get(bt, bt)
    ax.set_title(f"CDF-normalized performance: {label} / {task}")
    fig.tight_layout()
    save_figure(fig, output_dir / f"cdf_normalized_{bt}_{task}")


# ── Plot 3: Multi-file summary across building types ─────────────────


def plot_cross_type_summary(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    """Grouped bar chart: overall mean reward per (building_type, task)."""
    apply_paper_style()

    labels: list[str] = []
    means: list[float] = []
    errors: list[float] = []

    for s in sorted(
        summaries, key=lambda x: (x.get("building_type", ""), x.get("task", ""))
    ):
        bt = s.get("building_type", "?")
        task = s.get("task", "?")
        label = BUILDING_TYPE_LABELS.get(bt, bt)
        labels.append(f"{label}\n{task}")

        all_rewards = np.array(
            [r for b in s["buildings"].values() for r in b["rewards"]]
        )
        means.append(float(all_rewards.mean()) if len(all_rewards) > 0 else 0.0)
        if len(all_rewards) >= 2:
            lo, hi = _bootstrap_ci(all_rewards)
            errors.append((hi - lo) / 2)
        else:
            errors.append(0.0)

    width = max(len(labels) * 0.6, FIGSIZE_SINGLE[0])
    fig, ax = plt.subplots(figsize=(width, FIGSIZE_SINGLE[1]))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=errors,
        color=APPROACH_COLORS.get("ppo", "#DD8452"),
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean episode return")
    ax.set_title("CHS PPO re-evaluation summary")
    fig.tight_layout()
    save_figure(fig, output_dir / "chs_summary")


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CHS PPO re-evaluation results.")
    parser.add_argument(
        "--reeval-dir",
        type=Path,
        required=True,
        help="Directory containing reeval_*.yaml files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/chs"),
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    reeval_files = sorted(args.reeval_dir.glob("reeval_*.yaml"))
    if not reeval_files:
        print(f"No reeval_*.yaml files found in {args.reeval_dir}")
        return

    summaries: list[dict[str, Any]] = []
    for path in reeval_files:
        summary = _load_reeval(path)
        summaries.append(summary)
        plot_per_building_rewards(summary, args.output_dir)
        plot_cdf_normalized(summary, args.output_dir)

    if len(summaries) > 1:
        plot_cross_type_summary(summaries, args.output_dir)

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
