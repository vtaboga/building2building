from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
from gymnasium.spaces import Box, Dict

from b2b.make_env import make_env

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg):
    output_dir = Path.cwd()

    env = make_env(config=cfg, eplus_output_dir=str(output_dir / "eplus_outputs"))

    def _zeros_for_space(space) -> Any:
        if isinstance(space, Dict):
            return {k: _zeros_for_space(sub) for k, sub in space.spaces.items()}
        if isinstance(space, Box):
            return np.zeros(space.shape, dtype=float)
        # Fallback: a scalar zero
        return 0.0

    constant_action = _zeros_for_space(env.action_space)

    class _ConstantPolicy:
        def predict(self, obs, deterministic: bool = True):
            return constant_action, None

    policy = _ConstantPolicy()

    for ep in range(int(getattr(cfg, "n_episodes", 1))):
        obs, _info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
            steps += 1

        print(f"episode={ep + 1} steps={steps} total_reward={total_reward}", flush=True)

    env.close()

if __name__ == "__main__":
    main()

