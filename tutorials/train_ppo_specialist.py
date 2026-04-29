#!/usr/bin/env python3
"""Train a PPO specialist on a single building and evaluate it.

Demonstrates the full train-evaluate-score pipeline using the
building2building API and Stable-Baselines3.
"""

from __future__ import annotations

import building2building as b2b
from stable_baselines3 import PPO


def main() -> None:
    building_type = "OfficeSmall"
    task = "task1"
    run_period = "winter"
    total_timesteps = 50_000
    train_building_id = b2b.list_buildings(building_type, split="train")[0]
    eval_building_id = b2b.list_buildings(building_type, split="test")[0]

    # -- Train --
    print(
        f"Training PPO on {building_type}/{train_building_id}/{task} ({run_period})..."
    )
    train_env = b2b.new_make_env(
        building_type,
        building_id=train_building_id,
        task=task,
        run_period=run_period,
    )
    train_env = b2b.NormalizeObservation(train_env)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_office_small_winter")
    train_env.close()

    # -- Evaluate --
    print("\nEvaluating on test building...")
    eval_env = b2b.new_make_env(
        building_type,
        building_id=eval_building_id,
        task=task,
        run_period=run_period,
    )
    eval_env = b2b.NormalizeObservation(eval_env)

    model = PPO.load("ppo_office_small_winter")
    obs, _ = eval_env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Test episode return: {total_reward:.2f}")
    eval_env.close()

    # -- Score --
    try:
        score = b2b.compute_normalized_score(
            cumulative_return=total_reward,
            building_type=building_type,
            task=task,
            run_period=run_period,
            building_id=eval_building_id,
        )
        print(f"Normalized score: {score:.3f} (>1.0 = better than baseline)")
    except FileNotFoundError:
        print(
            "baseline_returns.csv not found. Run baselines.run_reactive_control first "
            "to generate it."
        )


if __name__ == "__main__":
    main()
