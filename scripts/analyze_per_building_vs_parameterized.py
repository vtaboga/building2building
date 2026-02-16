"""
Analysis script to compare per-building PPO specialists vs parameterized PPO.

Pulls runs from wandb and creates comparison plots:
1. Mean reward comparison across all test buildings
2. Standard deviation comparison across all test buildings
3. Per-building reward comparison (parameterized vs specialist)

Usage:
    python scripts/analyze_per_building_vs_parameterized.py \
        --project building2building \
        --entity pierre-luc-bacon-mila-org \
        --batch-id 8679110 \
        --output-dir outputs/analysis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_wandb_runs(
    project: str,
    entity: str | None,
    batch_id: str,
    max_per_building_runs: int = 10,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Fetch the latest parameterized PPO run and per-building runs from wandb.

    Returns:
        (parameterized_run, per_building_runs)
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        return None, []

    api = wandb.Api()

    # Fetch parameterized PPO run (latest with tag "algo:ppo")
    logger.info("Fetching parameterized PPO run...")
    filters = {"tags": {"$in": ["algo:ppo"]}}
    if entity:
        path = f"{entity}/{project}"
    else:
        path = project

    param_runs = api.runs(
        path=path,
        filters=filters,
        order="-created_at",  # Most recent first
    )

    parameterized_run = None
    for run in param_runs:
        # Check if this is a parameterized run (has building params augmented)
        if run.config.get("env", {}).get("augment_building_params", False):
            parameterized_run = run
            logger.info(f"Found parameterized run: {run.name} ({run.id})")
            break

    if parameterized_run is None:
        logger.warning("No parameterized PPO run found")

    # Fetch per-building runs with the specified batch ID
    logger.info(f"Fetching per-building runs with batch={batch_id}...")
    filters = {"tags": {"$in": [f"batch={batch_id}"]}}
    per_building_runs = list(
        api.runs(
            path=path,
            filters=filters,
            order="-created_at",
        )
    )[:max_per_building_runs]

    logger.info(f"Found {len(per_building_runs)} per-building runs")

    return parameterized_run, per_building_runs


def extract_final_eval_stats(run: Any) -> dict[str, float] | None:
    """Extract final evaluation statistics from a wandb run."""
    try:
        summary = run.summary
        stats = {
            "mean": summary.get("adaptive_dynamics/return_mean"),
            "std": summary.get("adaptive_dynamics/return_std"),
            "median": summary.get("adaptive_dynamics/return_median"),
            "min": summary.get("adaptive_dynamics/return_min"),
            "max": summary.get("adaptive_dynamics/return_max"),
            "n_envs": summary.get("adaptive_dynamics/return_n_envs"),
        }

        # Check if we have the required stats
        if stats["mean"] is None or stats["std"] is None:
            return None

        return stats
    except Exception as e:
        logger.warning(f"Failed to extract stats from run {run.id}: {e}")
        return None


def extract_per_building_rewards(run: Any) -> dict[int, float] | None:
    """
    Extract per-building rewards from a wandb run.

    For per-building runs, this is just the single building's reward.
    For parameterized runs, we need to fetch the full evaluation results.
    """
    try:
        # For now, we'll use the summary stats
        # TODO: Fetch detailed per-building results from artifacts/history
        summary = run.summary
        split_index = run.config.get("split_index")

        if split_index is not None:
            # Per-building run - single building
            mean_reward = summary.get("adaptive_dynamics/return_mean")
            if mean_reward is not None:
                return {int(split_index): float(mean_reward)}

        return None
    except Exception as e:
        logger.warning(f"Failed to extract per-building rewards from run {run.id}: {e}")
        return None


def plot_mean_reward_comparison(
    parameterized_stats: dict[str, float] | None,
    per_building_stats: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Plot comparison of mean rewards."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-building runs
    per_building_means = [s["mean"] for s in per_building_stats]
    x_positions = list(range(len(per_building_means)))

    ax.bar(
        x_positions,
        per_building_means,
        alpha=0.6,
        label="Per-building specialists",
        color="steelblue",
    )

    # Parameterized run (horizontal line)
    if parameterized_stats:
        ax.axhline(
            y=parameterized_stats["mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Parameterized PPO (mean={parameterized_stats['mean']:.0f})",
        )

    ax.set_xlabel("Per-building run index")
    ax.set_ylabel("Mean reward across test buildings")
    ax.set_title("Mean Reward Comparison: Per-building vs Parameterized PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved mean reward comparison to {output_path}")
    plt.close()


def plot_std_comparison(
    parameterized_stats: dict[str, float] | None,
    per_building_stats: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Plot comparison of reward standard deviations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-building runs
    per_building_stds = [s["std"] for s in per_building_stats]
    x_positions = list(range(len(per_building_stds)))

    ax.bar(
        x_positions,
        per_building_stds,
        alpha=0.6,
        label="Per-building specialists",
        color="steelblue",
    )

    # Parameterized run (horizontal line)
    if parameterized_stats:
        ax.axhline(
            y=parameterized_stats["std"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Parameterized PPO (std={parameterized_stats['std']:.0f})",
        )

    ax.set_xlabel("Per-building run index")
    ax.set_ylabel("Reward standard deviation across test buildings")
    ax.set_title("Reward Std Dev Comparison: Per-building vs Parameterized PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved std comparison to {output_path}")
    plt.close()


def plot_per_building_comparison(
    parameterized_rewards: dict[int, float],
    per_building_rewards: dict[int, float],
    output_path: Path,
) -> None:
    """
    Plot per-building reward comparison.

    Shows reward for each building when controlled by:
    - Its specialist (per-building PPO)
    - The generalist (parameterized PPO)
    """
    # Find common buildings
    common_buildings = sorted(
        set(parameterized_rewards.keys()) & set(per_building_rewards.keys())
    )

    if not common_buildings:
        logger.warning("No common buildings found for per-building comparison")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(common_buildings))
    width = 0.35

    param_values = [parameterized_rewards[b] for b in common_buildings]
    specialist_values = [per_building_rewards[b] for b in common_buildings]

    ax.bar(
        x - width / 2,
        specialist_values,
        width,
        label="Specialist (per-building)",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        param_values,
        width,
        label="Generalist (parameterized)",
        alpha=0.8,
    )

    ax.set_xlabel("Building index")
    ax.set_ylabel("Reward")
    ax.set_title("Per-Building Reward Comparison: Specialist vs Generalist")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in common_buildings])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved per-building comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-building vs parameterized PPO performance"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="building2building",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="pierre-luc-bacon-mila-org",
        help="WandB entity (username or team)",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        required=True,
        help="SLURM array job ID for per-building runs",
    )
    parser.add_argument(
        "--max-per-building-runs",
        type=int,
        default=10,
        help="Maximum number of per-building runs to fetch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch runs from wandb
    parameterized_run, per_building_runs = fetch_wandb_runs(
        project=args.project,
        entity=args.entity,
        batch_id=args.batch_id,
        max_per_building_runs=args.max_per_building_runs,
    )

    if not per_building_runs:
        logger.error("No per-building runs found. Exiting.")
        return

    # Extract statistics
    parameterized_stats = None
    if parameterized_run:
        parameterized_stats = extract_final_eval_stats(parameterized_run)

    per_building_stats = []
    per_building_rewards_map = {}
    for run in per_building_runs:
        stats = extract_final_eval_stats(run)
        if stats:
            per_building_stats.append(stats)

        # Extract per-building rewards
        rewards = extract_per_building_rewards(run)
        if rewards:
            per_building_rewards_map.update(rewards)

    if not per_building_stats:
        logger.error("No valid per-building statistics found. Exiting.")
        return

    logger.info(f"Extracted stats from {len(per_building_stats)} per-building runs")

    # Plot 1: Mean reward comparison
    plot_mean_reward_comparison(
        parameterized_stats,
        per_building_stats,
        args.output_dir / "mean_reward_comparison.png",
    )

    # Plot 2: Std comparison
    plot_std_comparison(
        parameterized_stats,
        per_building_stats,
        args.output_dir / "std_comparison.png",
    )

    # Plot 3: Per-building comparison (if we have parameterized per-building data)
    if parameterized_run and per_building_rewards_map:
        # TODO: Fetch detailed per-building results from parameterized run
        logger.warning(
            "Per-building comparison requires fetching detailed evaluation results. "
            "This feature is not yet implemented."
        )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
