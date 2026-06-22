#!/usr/bin/env python3
"""Evaluate dynamics adaptation models on held-out test buildings.

Loads a trained PPO model (specialist, baseline, or parameterized) and
evaluates on the test split of the dynamics adaptation benchmark.

Usage::

    python -m baselines.eval_dynamics_adaptation \
        --model-path outputs/dynamics_specialist/models/specialist_42.zip \
        --difficulty easy --approach specialist

    python -m baselines.eval_dynamics_adaptation \
        --model-path outputs/dynamics_parameterized/models/multi_parameterized.zip \
        --difficulty easy --approach parameterized
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import building2building as b2b
from baselines.utils.evaluation import EpisodeResult, run_episode

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    building_id: str
    reward: float
    episode_length: int
    normalized_score: float | None


def evaluate_specialist(
    model_dir: Path,
    bench: b2b.benchmarks.DynamicsAdaptation,
    n_episodes: int = 1,
) -> list[TestResult]:
    """Evaluate per-building specialist models on test buildings."""
    test_ids = bench.test_building_ids()
    results: list[TestResult] = []

    for bid in test_ids:
        model_path = model_dir / f"specialist_{bid}.zip"
        if not model_path.exists():
            logger.warning("No specialist model for %s, skipping", bid)
            continue

        model = PPO.load(str(model_path))
        for _ in range(n_episodes):
            env = b2b.make_env(
                bench.building_type, building_id=bid, task=bench.task
            )
            try:
                ep: EpisodeResult = run_episode(env, model)
                try:
                    ns = b2b.compute_normalized_score(
                        ep.total_reward,
                        bench.building_type,
                        bench.task,
                        run_period="full_year",
                        building_id=bid,
                    )
                except Exception:
                    ns = None
                results.append(
                    TestResult(
                        building_id=bid,
                        reward=ep.total_reward,
                        episode_length=ep.episode_length,
                        normalized_score=ns,
                    )
                )
                logger.info(
                    "  %s: reward=%.1f ns=%s",
                    bid,
                    ep.total_reward,
                    f"{ns:.4f}" if ns is not None else "N/A",
                )
            finally:
                env.close()

    return results


def evaluate_multi_building(
    model_path: Path,
    bench: b2b.benchmarks.DynamicsAdaptation,
    *,
    pad_obs_size: int,
    augment_params: bool = False,
    n_episodes: int = 1,
) -> list[TestResult]:
    """Evaluate a single multi-building model on all test buildings."""
    model = PPO.load(str(model_path))
    test_ids = bench.test_building_ids()
    results: list[TestResult] = []

    for bid in test_ids:
        for _ in range(n_episodes):
            env: Any = b2b.make_env(
                bench.building_type, building_id=bid, task=bench.task
            )
            env = b2b.PadObservation(env, target_size=pad_obs_size)
            env = b2b.wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)
            if augment_params:
                env = b2b.AugmentObservationWithBuildingParams(
                    env, allow_defaults=True
                )
            env = Monitor(env)
            try:
                ep: EpisodeResult = run_episode(env, model)
                try:
                    ns = b2b.compute_normalized_score(
                        ep.total_reward,
                        bench.building_type,
                        bench.task,
                        run_period="full_year",
                        building_id=bid,
                    )
                except Exception:
                    ns = None
                results.append(
                    TestResult(
                        building_id=bid,
                        reward=ep.total_reward,
                        episode_length=ep.episode_length,
                        normalized_score=ns,
                    )
                )
                logger.info(
                    "  %s: reward=%.1f ns=%s",
                    bid,
                    ep.total_reward,
                    f"{ns:.4f}" if ns is not None else "N/A",
                )
            finally:
                env.close()

    return results


def write_csv(results: list[TestResult], path: Path) -> None:
    fieldnames = [
        "building_id",
        "reward",
        "episode_length",
        "normalized_score",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "building_id": r.building_id,
                    "reward": f"{r.reward:.1f}",
                    "episode_length": r.episode_length,
                    "normalized_score": (
                        f"{r.normalized_score:.4f}"
                        if r.normalized_score is not None
                        else ""
                    ),
                }
            )
    logger.info("Wrote %d results to %s", len(results), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dynamics adaptation models")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=["specialist", "baseline", "parameterized"],
    )
    parser.add_argument("--task", type=str, default="task1")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(os.environ.get("SCRATCH", "outputs")),
        help="Root directory; results go under <base_dir>/b2b/eval/",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--pad-obs-size",
        type=int,
        default=None,
        help=(
            "Observation padding size used during training. "
            "If not given, loaded from metadata.json in the model directory."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    out_path = (
        args.output
        if args.output is not None
        else args.base_dir / "b2b" / "eval" / "dynamics_results.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bench = b2b.benchmarks.DynamicsAdaptation(
        difficulty=args.difficulty, task=args.task
    )
    logger.info(
        "Evaluating %s approach on %s (difficulty=%s, task=%s)",
        args.approach,
        bench.building_type,
        args.difficulty,
        args.task,
    )

    if args.approach == "specialist":
        model_dir = Path(args.model_path)
        results = evaluate_specialist(model_dir, bench, n_episodes=args.n_episodes)
    elif args.approach in ("baseline", "parameterized"):
        pad_obs_size: int | None = args.pad_obs_size
        if pad_obs_size is None:
            metadata_path = Path(args.model_path).parent / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"metadata.json not found at {metadata_path}. "
                    "Pass --pad-obs-size explicitly or re-run training to "
                    "generate the metadata file."
                )
            pad_obs_size = int(json.loads(metadata_path.read_text())["pad_obs_size"])
        logger.info("Using pad_obs_size=%d", pad_obs_size)
        results = evaluate_multi_building(
            Path(args.model_path),
            bench,
            pad_obs_size=pad_obs_size,
            augment_params=(args.approach == "parameterized"),
            n_episodes=args.n_episodes,
        )
    else:
        raise ValueError(f"Unknown approach: {args.approach!r}")

    write_csv(results, out_path)

    if results:
        scores = [r.normalized_score for r in results if r.normalized_score is not None]
        if scores:
            logger.info(
                "Mean normalized score: %.4f (std=%.4f, n=%d)",
                np.mean(scores),
                np.std(scores),
                len(scores),
            )


if __name__ == "__main__":
    main()
