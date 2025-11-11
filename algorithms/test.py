import csv
from pathlib import Path
from typing import Any, List

import numpy as np
from gymnasium.spaces import Box, Dict
from omegaconf import OmegaConf

from algorithms.utils import make_env
from building2building.simulator.wrappers import NormalizeObservation

import logging

logger = logging.getLogger(__name__)


def _flatten_by_space(sample: Any, space) -> List[float]:
    """Flatten a sample according to its Gymnasium space structure."""
    if isinstance(space, Dict):
        out: List[float] = []
        # Preserve deterministic order using space.spaces items order
        for k, sub in space.spaces.items():
            v = sample[k] if isinstance(sample, dict) else None
            out.extend(_flatten_by_space(v, sub))
        return out
    if isinstance(space, Box):
        size = int(np.prod(space.shape or (1,)))
        if sample is None:
            return [0.0] * size
        arr = np.asarray(sample, dtype=float).reshape(-1)
        return [float(x) for x in arr.tolist()]
    # Fallbacks
    if isinstance(sample, (list, tuple, np.ndarray)):
        return [float(x) for x in np.asarray(sample, dtype=float).reshape(-1).tolist()]
    try:
        return [float(sample)]
    except Exception:
        return []


def _names_from_space(space, prefix: str) -> List[str]:
    """Create simple column names from a space when explicit names are unavailable."""
    if isinstance(space, Dict):
        names: List[str] = []
        for k, sub in space.spaces.items():
            names.extend(_names_from_space(sub, f"{prefix}.{k}"))
        return names
    if isinstance(space, Box):
        size = int(np.prod(space.shape or (1,)))
        return [f"{prefix}_{i}" for i in range(size)]
    return [prefix]


def _get_names(env, kind: str, space, default_prefix: str) -> List[str]:
    """Try to get names from env metadata; otherwise derive from space shape."""
    # Prefer metadata when available
    try:
        if hasattr(env, "metadata") and isinstance(env.metadata, dict):
            meta_key = f"{kind}_names"
            if meta_key in env.metadata and isinstance(env.metadata[meta_key], list):
                return list(env.metadata[meta_key])
            # Backward-compat for observations
            if kind == "observation" and "observation_names" in env.metadata:
                return list(env.metadata["observation_names"])  # type: ignore
    except Exception:
        pass
    # Fallback to space-derived names
    return _names_from_space(space, default_prefix)


def test_policy(config, policy_model, output_dir: Path):
    """Run policy for n episodes, logging obs, actions, and reward to CSV."""

    test_dir = output_dir / "test"
    # Build a single env with a dedicated EnergyPlus output dir
    env = make_env(config=config, eplus_output_dir=str(test_dir / "eplus_outputs"))
    # Require explicit env.normalize_obs
    norm_obs = config.env.normalize_obs
    if norm_obs:
        env = NormalizeObservation(env)

    # Column names
    obs_names = _get_names(env, "observation", env.observation_space, "obs")
    act_names = _get_names(env, "action", env.action_space, "action")
    header = obs_names + act_names + ["reward"]

    # Output directory for CSV logs
    test_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(config.n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        rows: List[List[float]] = []

        while not done:
            action, _states = policy_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info  = env.step(action)

            denorm_obs = env.denormalize(obs)
            flat_obs = _flatten_by_space(denorm_obs, env.observation_space)
            flat_act = _flatten_by_space(action, env.action_space)
            rows.append(flat_obs + flat_act + [float(reward)])
            total_reward += float(reward)

            done = bool(terminated or truncated)

        csv_path = test_dir / f"policy_episode_{ep + 1}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        logger.info(f"Test episode {ep + 1} - steps: {len(rows)} - total reward: {total_reward}")

    try:
        env.close()
    except Exception:
        pass

