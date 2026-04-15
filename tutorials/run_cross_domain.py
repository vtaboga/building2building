#!/usr/bin/env python3
"""Cross-domain transfer exploration with the Amorpheus policy.

Demonstrates how the Amorpheus transformer handles buildings with
different observation and action dimensions via the morphology graph.
"""

from __future__ import annotations

import torch

import building2building as b2b
from baselines.models.amorpheus import AmorpheusPolicy


def main() -> None:
    bench = b2b.benchmarks.CrossDomainGeneralization(difficulty="easy", task="task1")
    print(f"Train type: {bench.train_type}")
    print(f"Test type:  {bench.test_type}")

    # -- Compare building types --
    for btype in [bench.train_type, bench.test_type]:
        env = b2b.new_make_env(btype, split="train", index=0, task="task1")
        morph = env.metadata["morphology"]
        print(
            f"\n{btype}: obs={env.observation_space.shape[0]}, "
            f"act={env.action_space.shape[0]}, "
            f"nodes={len(morph.nodes)}"
        )
        for node in morph.nodes:
            print(
                f"  {node.node_id}: type={node.node_type.name}, "
                f"obs_dim={node.node_type.observation_dim}, "
                f"act_dim={node.node_type.action_dim}"
            )
        env.close()

    # -- Instantiate Amorpheus --
    train_env = b2b.new_make_env(
        bench.train_type, split="train", index=0, task="task1"
    )
    train_morph = train_env.metadata["morphology"]

    policy = AmorpheusPolicy(
        morphology=train_morph, building_type=bench.train_type, d_model=64
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\nAmorpheus parameters: {n_params:,}")

    # -- Forward pass on train type --
    obs_train = torch.randn(1, train_env.observation_space.shape[0])
    dist_train, values_train = policy(obs_train)
    print(
        f"\n{bench.train_type}: "
        f"action_dist={dist_train}, values={values_train.shape}"
    )
    train_env.close()

    # -- Transfer to test type --
    test_env = b2b.new_make_env(
        bench.test_type, split="test", index=0, task="task1"
    )
    test_morph = test_env.metadata["morphology"]
    policy.morphology = test_morph
    policy.building_type = bench.test_type

    obs_test = torch.randn(1, test_env.observation_space.shape[0])
    dist_test, values_test = policy(obs_test)
    print(
        f"{bench.test_type}: "
        f"action_dist={dist_test}, values={values_test.shape}"
    )
    print("\nSame policy, different building type -- Amorpheus adapts via morphology!")
    test_env.close()


if __name__ == "__main__":
    main()
