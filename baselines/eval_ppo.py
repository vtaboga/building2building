#!/usr/bin/env python3
"""Evaluate trained PPO specialist models on their respective buildings.

Loads saved PPO models (from ``train_ppo.py``) and evaluates each on
its building/task, writing a CSV summary with normalized scores.

Usage::

    python -m baselines.eval_ppo \
        --model-dir outputs/train_ppo/2025-01-01/12-00-00 \
        --output results_ppo.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import building2building as b2b
from baselines.utils.evaluation import EpisodeResult, run_episode
from baselines.utils.training import make_rl_env_fn

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    building_type: str
    building_id: str
    task: str
    reward_mean: float
    episode_length: int
    normalized_score: float | None


def evaluate_model(
    model_path: Path,
    building_type: str,
    building_id: str,
    task: str,
    *,
    n_episodes: int = 1,
    run_period: str = "full_year",
) -> list[EvalResult]:
    model = PPO.load(str(model_path))
    results: list[EvalResult] = []

    for _ in range(n_episodes):
        env = make_rl_env_fn(
            building_type=building_type,
            building_id=building_id,
            task=task,
            run_period=run_period,
            normalize_obs=True,
            rescale_action=True,
            monitor=False,
        )()
        try:
            ep: EpisodeResult = run_episode(env, model)
            try:
                ns = b2b.compute_normalized_score(
                    ep.total_reward,
                    building_type,
                    task,
                    run_period=run_period,
                    building_id=building_id,
                )
            except Exception:
                ns = None
            results.append(
                EvalResult(
                    building_type=building_type,
                    building_id=building_id,
                    task=task,
                    reward_mean=ep.total_reward,
                    episode_length=ep.episode_length,
                    normalized_score=ns,
                )
            )
        finally:
            env.close()

    return results


def _parse_model_path(path: Path) -> tuple[str, str, str] | None:
    """Extract (building_type, building_id, task) from a nested model path.

    Expected directory structure:
    ``.../<output_dir>/models/<building_type>/<task>/ppo_<building_id>.zip``
    """
    if not path.stem.startswith("ppo_"):
        return None
    building_id = path.stem[4:]
    task = path.parent.name
    building_type = path.parent.parent.name
    if not building_id or not task or not building_type:
        return None
    return building_type, building_id, task


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO specialists")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing saved PPO models",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(os.environ.get("SCRATCH", "outputs")),
        help="Root directory; results go under <base_dir>/b2b/eval/",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--n-episodes", type=int, default=1, help="Episodes per model")
    parser.add_argument(
        "--run-period",
        type=str,
        default="full_year",
        choices=["full_year", "winter", "summer"],
        help="Simulation run period for evaluation and baseline lookup.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    out_path = (
        args.output
        if args.output is not None
        else args.base_dir / "b2b" / "eval" / "ppo_results.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    model_files = sorted(model_dir.rglob("ppo_*.zip"))
    if not model_files:
        logger.error("No model files found under %s", model_dir)
        return

    logger.info("Found %d model files", len(model_files))
    all_results: list[EvalResult] = []

    for mf in model_files:
        parsed = _parse_model_path(mf)
        if parsed is None:
            logger.warning("Skipping unrecognized file: %s", mf)
            continue
        bt, bid, task = parsed
        logger.info("Evaluating %s/%s task=%s", bt, bid, task)
        try:
            results = evaluate_model(
                mf,
                bt,
                bid,
                task,
                n_episodes=args.n_episodes,
                run_period=args.run_period,
            )
            all_results.extend(results)
        except Exception:
            logger.exception("Failed: %s", mf.name)

    fieldnames = [
        "building_type",
        "building_id",
        "task",
        "reward_mean",
        "episode_length",
        "normalized_score",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    "building_type": r.building_type,
                    "building_id": r.building_id,
                    "task": r.task,
                    "reward_mean": f"{r.reward_mean:.1f}",
                    "episode_length": r.episode_length,
                    "normalized_score": (
                        f"{r.normalized_score:.4f}"
                        if r.normalized_score is not None
                        else ""
                    ),
                }
            )

    logger.info("Wrote %d results to %s", len(all_results), out_path)
    if all_results:
        scores = [
            r.normalized_score for r in all_results if r.normalized_score is not None
        ]
        if scores:
            logger.info(
                "Mean normalized score: %.4f (std=%.4f)",
                np.mean(scores),
                np.std(scores),
            )


if __name__ == "__main__":
    main()
