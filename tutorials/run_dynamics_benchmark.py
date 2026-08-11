#!/usr/bin/env python3
"""Mini dynamics adaptation experiment.

Trains a multi-building PPO with building parameter augmentation on
the 'easy' difficulty benchmark (SingleFamilyHouse), then evaluates
on held-out test buildings.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import building2building as b2b

N_TRAIN_BUILDINGS = 5
N_TEST_BUILDINGS = 3
TOTAL_TIMESTEPS = 50_000
PAD_SIZE = 20
RUN_PERIOD = "winter"


def main() -> None:
    bench = b2b.benchmarks.DynamicsAdaptation(difficulty="easy", task="task_const_e0")
    train_ids = bench.train_building_ids()[:N_TRAIN_BUILDINGS]
    test_ids = bench.test_building_ids()[:N_TEST_BUILDINGS]

    print(f"Building type: {bench.building_type}")
    print(f"Using {len(train_ids)} train buildings, {len(test_ids)} test buildings")

    # -- Build multi-building training environment --
    # Wrapper order matches baselines.train_dynamics_adaptation:
    # PadObservation -> wrap_env_for_rl (obs normalization + action
    # rescaling) -> AugmentObservationWithBuildingParams.
    def make_env(idx: int) -> gym.Env:
        env = b2b.make_env(
            bench.building_type,
            building_id=train_ids[idx],
            task="task_const_e0",
            run_period=RUN_PERIOD,
        )
        env = b2b.PadObservation(env, target_size=PAD_SIZE)
        env = b2b.wrap_env_for_rl(env, normalize_obs=True, rescale_action=True)
        env = b2b.AugmentObservationWithBuildingParams(env, allow_defaults=True)
        return env

    env = b2b.ResampleBuildingOnResetWrapper(
        make_env,
        available_indices=list(range(len(train_ids))),
    )

    # -- Train --
    print(f"\nTraining PPO for {TOTAL_TIMESTEPS} steps...")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_dynamics_easy")
    env.close()

    # -- Evaluate --
    print(f"\nEvaluating on {len(test_ids)} test buildings...")
    returns: list[float] = []

    for bid in test_ids:
        eval_env = b2b.make_env(
            bench.building_type,
            building_id=bid,
            task="task_const_e0",
            run_period=RUN_PERIOD,
        )
        eval_env = b2b.PadObservation(eval_env, target_size=PAD_SIZE)
        eval_env = b2b.wrap_env_for_rl(
            eval_env, normalize_obs=True, rescale_action=True
        )
        eval_env = b2b.AugmentObservationWithBuildingParams(
            eval_env, allow_defaults=True
        )

        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)
        print(f"  {bid}: return={total_reward:.0f}")
        eval_env.close()

    print(f"\nTest results: mean={np.mean(returns):.0f} +/- {np.std(returns):.0f}")


if __name__ == "__main__":
    main()
