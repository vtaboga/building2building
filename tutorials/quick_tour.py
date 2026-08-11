#!/usr/bin/env python3
"""Quick tour of the Building2Building API.

Creates environments, inspects observation/action spaces, runs a random-action
episode, and explores the morphology graph.
"""

from __future__ import annotations

import building2building as b2b


def main() -> None:
    # -- List available building types --
    types = b2b.list_building_types()
    print(f"Available building types ({len(types)}): {types}")

    # -- Create an environment --
    env = b2b.make_env("OfficeSmall", split="test", index=0, task="task_const_e0")
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    # -- Inspect metadata --
    obs_names = env.metadata["observation_names"]
    act_names = env.metadata["action_names"]
    equipment = env.metadata["hvac_equipment"]

    print(f"\nObservation names ({len(obs_names)}):")
    for i, name in enumerate(obs_names):
        print(f"  [{i}] {name}")

    print(f"\nAction names ({len(act_names)}):")
    for i, name in enumerate(act_names):
        print(f"  [{i}] {name}")

    print(f"\nHVAC equipment ({len(equipment)} systems):")
    for eq in equipment:
        print(f"  {type(eq).__name__}: zones={eq.zones()}")

    # -- Run a random-action episode --
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if steps <= 3:
            print(f"\nStep {steps}: reward={reward:.2f}, obs[:5]={obs[:5]}")

    print(f"\nEpisode finished: {steps} steps, return={total_reward:.2f}")
    env.close()

    # -- Explore the morphology graph --
    env2 = b2b.make_env("OfficeSmall", split="test", index=0, task="task_const_e0")
    morph = env2.metadata["morphology"]

    print(f"\n--- Morphology Graph ---")
    print(f"Nodes: {len(morph.nodes)}")
    print(f"Edges: {len(morph.edges)}")
    print(f"Type counts: {morph.type_counts()}")

    obs2, _ = env2.reset()
    local_obs = morph.split_observation(obs2)
    print("\nPer-node local observations:")
    for node_id, local in local_obs.items():
        print(f"  {node_id}: shape={local.shape}")

    env2.close()

    # -- Compare building types --
    print("\n--- Building Type Comparison ---")
    for btype in ["SingleFamilyHouse", "OfficeSmall", "OfficeMedium"]:
        env_tmp = b2b.make_env(btype, split="test", index=0, task="task_const_e0")
        morph_tmp = env_tmp.metadata["morphology"]
        print(
            f"{btype}: obs={env_tmp.observation_space.shape[0]}, "
            f"act={env_tmp.action_space.shape[0]}, "
            f"nodes={len(morph_tmp.nodes)}, "
            f"types={morph_tmp.type_counts()}"
        )
        env_tmp.close()


if __name__ == "__main__":
    main()
