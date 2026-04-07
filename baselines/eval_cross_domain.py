#!/usr/bin/env python3
"""Evaluate a trained Amorpheus policy on cross-domain test buildings.

Loads a saved ``AmorpheusPolicy`` state dict and evaluates zero-shot
transfer on test buildings from the cross-domain benchmark.

Usage::

    python -m baselines.eval_cross_domain \
        --model-path outputs/cross_domain/amorpheus_policy.pt \
        --test-building-types Warehouse SingleFamilyHouse \
        --task task3 --n-test 5
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import building2building as b2b
from baselines.models.amorpheus import AmorpheusPolicy
from baselines.utils.evaluation import EpisodeResult, run_episode

logger = logging.getLogger(__name__)


@dataclass
class CrossDomainResult:
    building_type: str
    building_index: int
    reward: float
    episode_length: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Amorpheus cross-domain transfer"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--test-building-types",
        type=str,
        nargs="+",
        required=True,
        help="Building types to evaluate on",
    )
    parser.add_argument("--task", type=str, default="task3")
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--output", type=str, default="results_cross_domain.csv"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    ref_env = b2b.new_make_env(
        args.test_building_types[0], split="test", index=0, task=args.task
    )
    ref_morph: b2b.Morphology = ref_env.metadata["morphology"]
    ref_env.close()

    policy = AmorpheusPolicy(
        morphology=ref_morph,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    logger.info("Loaded policy from %s", args.model_path)

    results: list[CrossDomainResult] = []

    for bt in args.test_building_types:
        logger.info("Testing on %s (%d buildings)", bt, args.n_test)
        for i in range(args.n_test):
            try:
                env = b2b.new_make_env(
                    bt, split="test", index=i, task=args.task
                )
                morph = env.metadata["morphology"]
                policy.morphology = morph

                ep: EpisodeResult = run_episode(env, policy)
                results.append(
                    CrossDomainResult(
                        building_type=bt,
                        building_index=i,
                        reward=ep.total_reward,
                        episode_length=ep.episode_length,
                    )
                )
                logger.info(
                    "  %s[%d]: reward=%.1f  len=%d",
                    bt,
                    i,
                    ep.total_reward,
                    ep.episode_length,
                )
                env.close()
            except Exception:
                logger.exception("Failed: %s[%d]", bt, i)

    out_path = Path(args.output)
    fieldnames = [
        "building_type",
        "building_index",
        "reward",
        "episode_length",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "building_type": r.building_type,
                    "building_index": r.building_index,
                    "reward": f"{r.reward:.1f}",
                    "episode_length": r.episode_length,
                }
            )

    logger.info("Wrote %d results to %s", len(results), out_path)

    for bt in args.test_building_types:
        bt_results = [r for r in results if r.building_type == bt]
        if bt_results:
            rewards = [r.reward for r in bt_results]
            logger.info(
                "%s: mean=%.1f  std=%.1f  n=%d",
                bt,
                np.mean(rewards),
                np.std(rewards),
                len(rewards),
            )


if __name__ == "__main__":
    main()
