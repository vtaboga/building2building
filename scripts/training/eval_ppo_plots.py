#!/usr/bin/env python3
"""Run short PPO rollouts (2016 steps ≈ 3 weeks) and generate diagnostic plots.

For each gathered model, runs a 2016-step deterministic rollout and saves
``rollout.csv`` plus diagnostic plots (temperature, actuators, energy, reward)
into per-case subdirectories::

    <output-dir>/
      <BuildingType>/
        <task>/
          building_<idx>/
            rollout.csv
            temperature.png
            actuators_*.png
            energy.png
            reward.png

Usage:
    python scripts/eval_ppo_plots.py
    python scripts/eval_ppo_plots.py --output-dir outputs/eval_ppo_plots --max-steps 2016
"""

from __future__ import annotations

import argparse
import importlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from building2building.api import make_multizones_env
from building2building.benchmark.plot_rollout import plot_rollout_df
from building2building.benchmark.runner import run_rollout
from building2building.env import setup_energyplus_path
from building2building.simulator.wrappers import NormalizeObservation
from building2building.sources.multizones_reference_buildings import (
    building_id_from_split_index,
    climate_zone_for_building,
)
from building2building.types import RunPeriodConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 2016

DEFAULT_MODELS_DIR = Path(
    "/network/scratch/v/vincent.taboga/Building2Building/trained_models"
)


def _load_sb3_model(algo_name: str, model_path: Path) -> Any:
    algo_upper = str(algo_name).upper()
    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(
                f"{base_pkg}.{algo_name}.{algo_name}"
            )
            algo_cls = getattr(module, algo_upper)
            model = algo_cls.load(str(model_path))
            log.info("Loaded model from %s", model_path)
            return model
        except (ModuleNotFoundError, AttributeError):
            continue
    raise RuntimeError(f"Cannot load {algo_name} model from {model_path}")


@dataclass
class ModelDir:
    building_type: str
    task: str
    building_index: int
    model_path: Path
    config_path: Path


def discover_models(models_dir: Path) -> list[ModelDir]:
    """Find all gathered model directories."""
    found: list[ModelDir] = []
    for bt_dir in sorted(models_dir.iterdir()):
        if not bt_dir.is_dir():
            continue
        for task_dir in sorted(bt_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for bldg_dir in sorted(task_dir.iterdir()):
                if not bldg_dir.is_dir():
                    continue
                model_path = bldg_dir / "model.zip"
                config_path = bldg_dir / "config.yaml"
                if not model_path.exists():
                    continue
                idx_str = bldg_dir.name.replace("building_", "")
                if not idx_str.isdigit():
                    continue
                found.append(
                    ModelDir(
                        building_type=bt_dir.name,
                        task=task_dir.name,
                        building_index=int(idx_str),
                        model_path=model_path,
                        config_path=config_path,
                    )
                )
    return found


def eval_and_plot(
    md: ModelDir,
    output_dir: Path,
    eplus_base_dir: Path,
    max_steps: int,
) -> Path:
    """Run a short rollout and generate diagnostic plots for one model."""
    if not md.config_path.exists():
        raise FileNotFoundError(f"No config at {md.config_path}")

    cfg = OmegaConf.load(md.config_path)
    building_type = str(cfg.bldg.building_type)
    split = str(cfg.bldg.split)
    index = int(cfg.bldg.index)
    algo = str(cfg.policy.algorithm)

    norm_obs = bool(getattr(cfg.env, "normalize_obs", False))
    norm_action = bool(getattr(cfg.env, "normalize_action", False))

    task_dict: dict[str, Any] = OmegaConf.to_container(cfg.task, resolve=True)  # type: ignore[assignment]
    reward_dict: dict[str, Any] = OmegaConf.to_container(cfg.reward, resolve=True)  # type: ignore[assignment]

    building_id = building_id_from_split_index(building_type, split, index)
    cz = climate_zone_for_building(building_type, building_id)

    case_dir = output_dir / building_type / md.task / f"building_{index}"
    case_dir.mkdir(parents=True, exist_ok=True)

    eplus_dir = eplus_base_dir / building_type / md.task / f"building_{index}"
    eplus_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Evaluating %s/%s/idx=%d (bid=%d, CZ%d)  max_steps=%d",
        building_type,
        md.task,
        index,
        building_id,
        cz,
        max_steps,
    )

    env = make_multizones_env(
        building_type=building_type,
        split=split,
        split_index=index,
        eplus_output_dir=str(eplus_dir),
        task=task_dict,
        reward=reward_dict,
        max_steps=max_steps,
    )

    try:
        obs_names: list[str] = env.metadata["observation_names"]
        act_names: list[str] = env.metadata["action_names"]

        raw_obs_low = env.observation_space.low.copy()
        raw_obs_range = (
            env.observation_space.high - env.observation_space.low
        ).copy()
        raw_act_low = env.action_space.low.copy()
        raw_act_high = env.action_space.high.copy()

        if norm_action:
            env = gym.wrappers.RescaleAction(
                env, min_action=-1.0, max_action=1.0
            )
        if norm_obs:
            env = NormalizeObservation(env)

        model = _load_sb3_model(algo, md.model_path)

        results, data = run_rollout(
            env=env,
            policy=model,
            n_episodes=1,
            deterministic=True,
            max_steps=max_steps,
            record=True,
        )
        assert data is not None

        obs_raw = data.obs
        if norm_obs:
            obs_raw = (
                obs_raw * raw_obs_range[np.newaxis, :]
                + raw_obs_low[np.newaxis, :]
            )

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
        csv_path = case_dir / "rollout.csv"
        df.to_csv(csv_path, index=False)

        target_c = float(
            getattr(
                getattr(cfg.task, "default_zone_target_temperature", None),
                "occupied_c",
                21.0,
            )
        )
        plots = plot_rollout_df(df, case_dir, target_c=target_c)

        log.info(
            "  return=%.1f  steps=%d  plots=%d  → %s",
            results[0].total_reward,
            results[0].n_steps,
            len(plots),
            case_dir,
        )
        return case_dir
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root of gathered models (from gather_trained_models.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_ppo_plots"),
        help="Base output directory for rollout CSVs and plots.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Number of rollout steps per model (default: 2016 ≈ 3 weeks).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate all models, ignoring existing plots.",
    )
    args = parser.parse_args()

    setup_energyplus_path()

    models = discover_models(args.models_dir)
    log.info("Found %d models to evaluate", len(models))

    if not models:
        log.warning("No models found in %s — exiting.", args.models_dir)
        return

    if args.force:
        pending = models
    else:
        pending = []
        for md in models:
            case_dir = (
                args.output_dir
                / md.building_type
                / md.task
                / f"building_{md.building_index}"
            )
            if (case_dir / "rollout.csv").exists():
                continue
            pending.append(md)
        log.info(
            "%d models already have plots, %d remaining",
            len(models) - len(pending),
            len(pending),
        )

    if not pending:
        log.info("Nothing new to evaluate.")
        return

    eplus_base_dir = Path(tempfile.mkdtemp(prefix="eval_ppo_plots_"))
    log.info("EnergyPlus scratch dir: %s", eplus_base_dir)

    succeeded = 0
    failed = 0
    for i, md in enumerate(pending):
        log.info(
            "[%d/%d] %s / %s / building_%d",
            i + 1,
            len(pending),
            md.building_type,
            md.task,
            md.building_index,
        )
        try:
            eval_and_plot(md, args.output_dir, eplus_base_dir, args.max_steps)
            succeeded += 1
        except Exception:
            log.exception(
                "FAILED: %s/%s/building_%d",
                md.building_type,
                md.task,
                md.building_index,
            )
            failed += 1

    log.info(
        "\n%s\nDone: %d succeeded, %d failed out of %d pending (%d already done)\n%s",
        "=" * 70,
        succeeded,
        failed,
        len(pending),
        len(models) - len(pending),
        "=" * 70,
    )


if __name__ == "__main__":
    main()
