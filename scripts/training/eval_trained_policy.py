#!/usr/bin/env python3
"""Evaluate a trained SB3 policy on its building and generate rollout plots.

Loads the best (or final) model from a training run directory, runs a
deterministic rollout, saves the CSV + plots to an output directory in the
home workspace for easy access.

Usage:
    python scripts/eval_trained_policy.py \\
        --run-dir $SCRATCH/Building2Building/outputs/train_multizones/OfficeSmall/test_8/ppo/20260228_083418 \\
        --output-dir outputs/eval/OfficeSmall_test_8

    # Evaluate all completed runs at once:
    python scripts/eval_trained_policy.py \\
        --run-dir $SCRATCH/.../test_0/ppo/<timestamp> \\
                  $SCRATCH/.../test_4/ppo/<timestamp> \\
        --output-dir outputs/eval
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

from b2b.api import make_multizones_env
from b2b.benchmark.plot_rollout import plot_rollout_df
from b2b.benchmark.runner import run_rollout
from b2b.env import setup_energyplus_path
from b2b.simulator.wrappers import NormalizeObservation
from b2b.types import RunPeriodConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_sb3_model(algo_name: str, model_path: Path) -> Any:
    algo_upper = str(algo_name).upper()
    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algo_name}.{algo_name}")
            algo_cls = getattr(module, algo_upper)
            model = algo_cls.load(str(model_path))
            log.info("Loaded model from %s", model_path)
            return model
        except (ModuleNotFoundError, AttributeError):
            continue
    raise RuntimeError(f"Cannot load {algo_name} model from {model_path}")


def _pick_model(run_dir: Path) -> Path:
    models_dir = run_dir / "models"
    for name in ("best_model.zip", "final_model.zip", "model.zip"):
        p = models_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No model found in {models_dir}")


def evaluate_run(
    run_dir: Path,
    output_dir: Path,
    *,
    run_period: str | None = None,
    max_steps: int | None = None,
) -> Path:
    """Evaluate a single training run and save results.

    Returns the output directory with CSV + plots.
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config at {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    building_type = str(cfg.bldg.building_type)
    split = str(cfg.bldg.split)
    index = int(cfg.bldg.index)
    algo = str(cfg.policy.algorithm)

    rp = run_period or str(getattr(cfg.task, "run_period", "full_year"))
    if max_steps is None:
        max_steps = int(
            getattr(
                cfg.env, "max_steps",
                RunPeriodConfig.from_name(rp).expected_steps(),
            )
        )

    norm_obs = bool(getattr(cfg.env, "normalize_obs", False))
    norm_action = bool(getattr(cfg.env, "normalize_action", False))

    log.info(
        "Evaluating %s/%s/index=%d  algo=%s  run_period=%s  max_steps=%d  "
        "norm_obs=%s  norm_action=%s",
        building_type, split, index, algo, rp, max_steps, norm_obs, norm_action,
    )

    model_path = _pick_model(run_dir)
    model = _load_sb3_model(algo, model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    eplus_dir = output_dir / "eplus_outputs"

    env = make_multizones_env(
        building_type=building_type,
        split=split,
        split_index=index,
        eplus_output_dir=str(eplus_dir),
        task={"run_period": rp},
        reward={"reward_type": "DeadbandRewardConfig", "energy_weight": 0.01, "dT": 1.0},
        max_steps=max_steps,
    )

    obs_names = env.metadata["observation_names"]
    act_names = env.metadata["action_names"]

    # Keep a reference to the unwrapped obs bounds for denormalization
    raw_obs_low = env.observation_space.low.copy()
    raw_obs_range = (env.observation_space.high - env.observation_space.low).copy()

    # Keep the original action bounds before RescaleAction maps them to [-1, 1]
    raw_act_low = env.action_space.low.copy()
    raw_act_high = env.action_space.high.copy()

    if norm_action:
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)

    if norm_obs:
        env = NormalizeObservation(env)

    results, data = run_rollout(
        env=env,
        policy=model,
        n_episodes=1,
        deterministic=True,
        max_steps=max_steps,
        record=True,
    )
    env.close()
    assert data is not None

    # Denormalize observations for plotting
    obs_raw = data.obs
    if norm_obs:
        obs_raw = obs_raw * raw_obs_range[np.newaxis, :] + raw_obs_low[np.newaxis, :]

    # Denormalize actions: RescaleAction maps [low, high] -> [-1, 1],
    # so invert: action_real = low + (action_norm + 1) * (high - low) / 2
    act_raw = data.action
    if norm_action:
        act_raw = raw_act_low[np.newaxis, :] + (act_raw + 1.0) * 0.5 * (
            raw_act_high[np.newaxis, :] - raw_act_low[np.newaxis, :]
        )

    df_dict: dict[str, Any] = {
        "episode": data.episode,
        "step": data.step,
        "global_step": (
            data.episode.astype(np.int64) * max_steps
            + data.step.astype(np.int64)
        ),
        "reward": data.reward,
    }
    for k, arr in data.metrics.items():
        df_dict[str(k)] = np.asarray(arr, dtype=float)
    for i, name in enumerate(obs_names):
        if i < obs_raw.shape[1]:
            df_dict[f"obs::{name}"] = obs_raw[:, i]
    for i, name in enumerate(act_names):
        if i < act_raw.shape[1]:
            df_dict[f"act::{name}"] = act_raw[:, i]

    df = pd.DataFrame(df_dict)
    csv_path = output_dir / "rollout.csv"
    df.to_csv(csv_path, index=False)

    target_c = float(getattr(
        getattr(cfg.task, "default_zone_target_temperature", None),
        "occupied_c", 21.0,
    ))
    plots = plot_rollout_df(df, output_dir, target_c=target_c)

    meta = {
        "building_type": building_type,
        "split": split,
        "index": index,
        "algorithm": algo,
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "run_period": rp,
        "max_steps": max_steps,
        "total_reward": float(results[0].total_reward),
        "n_steps": results[0].n_steps,
        "plots": [str(p) for p in plots],
    }
    with open(output_dir / "eval_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        "Done: %s  reward=%.1f  steps=%d  plots=%d  → %s",
        building_type, results[0].total_reward, results[0].n_steps,
        len(plots), output_dir,
    )

    # Print temperature summary for conditioned zones
    temp_cols = [
        c for c in df.columns
        if c.lower().startswith("obs::zone air temperature")
        and "attic" not in c.lower()
    ]
    if temp_cols:
        temps = df[temp_cols].values.flatten()
        log.info(
            "Zone temps: mean=%.1f°C  std=%.1f°C  min=%.1f°C  max=%.1f°C  "
            "in [19,23]°C=%.1f%%",
            temps.mean(), temps.std(), temps.min(), temps.max(),
            ((temps >= 19) & (temps <= 23)).mean() * 100,
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained SB3 policies and generate plots."
    )
    parser.add_argument(
        "--run-dir", type=Path, nargs="+", required=True,
        help="One or more training run directories (containing .hydra/ and models/).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/eval"),
        help="Base output directory for results.",
    )
    parser.add_argument("--run-period", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    setup_energyplus_path()

    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        if not (run_dir / ".hydra" / "config.yaml").exists():
            log.warning("Skipping %s — no .hydra/config.yaml", run_dir)
            continue

        cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
        bt = str(cfg.bldg.building_type)
        idx = int(cfg.bldg.index)
        algo = str(cfg.policy.algorithm)
        sub = f"{bt}_test_{idx}_{algo}"

        out = args.output_dir / sub
        try:
            evaluate_run(
                run_dir, out,
                run_period=args.run_period,
                max_steps=args.max_steps,
            )
        except Exception:
            log.exception("Failed to evaluate %s", run_dir)


if __name__ == "__main__":
    main()
