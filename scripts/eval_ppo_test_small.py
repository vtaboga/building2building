#!/usr/bin/env python3
"""Evaluate trained PPO models on test_small buildings and compute normalised scores.

Iterates over all gathered models in ``trained_models/<BuildingType>/<task>/building_<idx>/``,
runs a full-year deterministic rollout, records metrics, and optionally normalises
returns against the G36 baseline.

Usage:
    python scripts/eval_ppo_test_small.py
    python scripts/eval_ppo_test_small.py --output-dir outputs/eval_ppo
    python scripts/eval_ppo_test_small.py --models-dir /path/to/trained_models
"""

from __future__ import annotations

import argparse
import csv
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

from b2b.api import make_multizones_env
from b2b.benchmark.runner import run_rollout
from b2b.env import setup_energyplus_path
from b2b.simulator.wrappers import NormalizeObservation
from b2b.sources.multizones_reference_buildings import (
    BuildingType,
    building_id_from_split_index,
    climate_zone_for_building,
)
from b2b.types import RunPeriodConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEATING_SP = 20.0
COOLING_SP = 22.0

BUILDING_TYPES: list[BuildingType] = [
    "Warehouse",
    "RetailStandalone",
    "RestaurantFastFood",
    "OfficeMedium",
    "OfficeSmall",
]

TASKS = [
    "deadband_ew001_const",
    "deadband_ew001_occ",
    "deadband_ew01_const",
    "barrier_ew1",
]

CSV_FIELDS = [
    "building_type",
    "task",
    "building_index",
    "building_id",
    "climate_zone",
    "mean_pct_in_band",
    "worst_zone_pct_in_band",
    "episode_return",
    "n_steps",
    "g36_return",
    "normalized_return",
]

DEFAULT_MODELS_DIR = Path(
    "/network/scratch/v/vincent.taboga/Building2Building/trained_models"
)
DEFAULT_G36_DIR = Path("outputs/eval_g36")


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


def _zone_temp_indices(
    obs_names: list[str], controlled_zones: list[str]
) -> list[int]:
    prefix = "zone air temperature"
    indices: list[int] = []
    for zone in controlled_zones:
        zn = zone.strip().lower()
        for i, name in enumerate(obs_names):
            nl = name.strip().lower()
            if nl.startswith(prefix):
                zone_part = nl[len(prefix) :].strip()
                if zone_part == zn or zn in zone_part or zone_part in zn:
                    indices.append(i)
                    break
    return indices


def _zone_target_temp_indices(
    obs_names: list[str], controlled_zones: list[str]
) -> list[int]:
    prefix = "target_temperature"
    indices: list[int] = []
    for zone in controlled_zones:
        zn = zone.strip().lower()
        for i, name in enumerate(obs_names):
            nl = name.strip().lower()
            if nl.startswith(prefix):
                zone_part = nl[len(prefix) :].strip()
                if zone_part == zn or zn in zone_part or zone_part in zn:
                    indices.append(i)
                    break
    return indices


def compute_pct_in_band(
    obs: np.ndarray,
    temp_indices: list[int],
    low: float = HEATING_SP,
    high: float = COOLING_SP,
    target_indices: list[int] | None = None,
    dT: float = 1.0,
) -> tuple[float, float]:
    """Compute per-zone % of timesteps inside the comfort band."""
    if not temp_indices:
        return 0.0, 0.0
    pcts: list[float] = []
    for zi, idx in enumerate(temp_indices):
        temps = obs[:, idx]
        n = len(temps)
        if n == 0:
            continue
        if target_indices and zi < len(target_indices):
            targets = obs[:, target_indices[zi]]
            in_band = (temps >= targets - dT) & (temps <= targets + dT)
        else:
            in_band = (temps >= low) & (temps <= high)
        pcts.append(float(np.sum(in_band) / n * 100))
    if not pcts:
        return 0.0, 0.0
    return float(np.mean(pcts)), float(np.min(pcts))


def _reconstruct_task_reward_from_config(
    cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build task and reward dicts from a saved Hydra config."""
    reward_dict: dict[str, Any] = OmegaConf.to_container(cfg.reward, resolve=True)  # type: ignore[assignment]
    task_dict: dict[str, Any] = OmegaConf.to_container(cfg.task, resolve=True)  # type: ignore[assignment]
    return task_dict, reward_dict


def load_g36_baselines(
    g36_dir: Path,
) -> dict[tuple[str, str, int], float]:
    """Load G36 baseline returns keyed by (building_type, task, building_id)."""
    baselines: dict[tuple[str, str, int], float] = {}
    for bt_dir in g36_dir.iterdir():
        if not bt_dir.is_dir():
            continue
        building_type = bt_dir.name
        for task_dir in bt_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            csv_path = task_dir / "results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                bid = int(row["building_id"])
                ret = float(row["episode_return"])
                baselines[(building_type, task, bid)] = ret
    return baselines


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


def eval_one_model(
    md: ModelDir,
    eplus_base_dir: Path,
) -> dict[str, Any]:
    """Run a single PPO model rollout and return a results row."""
    if not md.config_path.exists():
        raise FileNotFoundError(f"No config at {md.config_path}")

    cfg = OmegaConf.load(md.config_path)
    building_type = str(cfg.bldg.building_type)
    split = str(cfg.bldg.split)
    index = int(cfg.bldg.index)
    algo = str(cfg.policy.algorithm)

    rp = str(getattr(cfg.task, "run_period", "full_year"))
    max_steps = int(
        getattr(
            cfg.env,
            "max_steps",
            RunPeriodConfig.from_name(rp).expected_steps(),
        )
    )
    norm_obs = bool(getattr(cfg.env, "normalize_obs", False))
    norm_action = bool(getattr(cfg.env, "normalize_action", False))

    task_dict, reward_dict = _reconstruct_task_reward_from_config(cfg)

    building_id = building_id_from_split_index(building_type, split, index)
    cz = climate_zone_for_building(building_type, building_id)

    eplus_dir = (
        eplus_base_dir
        / building_type
        / md.task
        / f"building_{index}"
    )
    eplus_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Evaluating %s/%s/idx=%d (bid=%d, CZ%d)  algo=%s",
        building_type,
        md.task,
        index,
        building_id,
        cz,
        algo,
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
        meta = env.metadata
        obs_names: list[str] = meta["observation_names"]
        controlled_zones: list[str] = meta.get("controlled_zones", [])
        temp_indices = _zone_temp_indices(obs_names, controlled_zones)
        target_indices = _zone_target_temp_indices(obs_names, controlled_zones)

        raw_obs_low = env.observation_space.low.copy()
        raw_obs_range = (
            env.observation_space.high - env.observation_space.low
        ).copy()

        if norm_action:
            env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
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

        mean_pct, worst_pct = compute_pct_in_band(
            obs_raw,
            temp_indices,
            target_indices=target_indices or None,
        )

        return {
            "building_type": building_type,
            "task": md.task,
            "building_index": index,
            "building_id": building_id,
            "climate_zone": cz,
            "mean_pct_in_band": round(mean_pct, 2),
            "worst_zone_pct_in_band": round(worst_pct, 2),
            "episode_return": round(results[0].total_reward, 1),
            "n_steps": results[0].n_steps,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _load_existing_results(
    csv_path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, int]]]:
    """Load previously computed rows and return them with a set of done keys."""
    if not csv_path.exists():
        return [], set()
    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    done: set[tuple[str, str, int]] = set()
    for row in rows:
        done.add((str(row["building_type"]), str(row["task"]), int(row["building_index"])))
    return rows, done


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
        default=Path("outputs/eval_ppo"),
        help="Directory for results CSV.",
    )
    parser.add_argument(
        "--g36-dir",
        type=Path,
        default=DEFAULT_G36_DIR,
        help="Root of G36 baseline eval results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate all models, ignoring existing results.",
    )
    args = parser.parse_args()

    setup_energyplus_path()

    models = discover_models(args.models_dir)
    log.info("Found %d models to evaluate", len(models))

    if not models:
        log.warning("No models found in %s — exiting.", args.models_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"

    if args.force:
        existing_rows: list[dict[str, Any]] = []
        done: set[tuple[str, str, int]] = set()
    else:
        existing_rows, done = _load_existing_results(csv_path)
        if done:
            log.info("Loaded %d existing results — will skip those.", len(done))

    pending = [
        md for md in models
        if (md.building_type, md.task, md.building_index) not in done
    ]
    log.info(
        "%d models already evaluated, %d remaining",
        len(models) - len(pending),
        len(pending),
    )

    if not pending:
        log.info("Nothing new to evaluate.")
        return

    eplus_base_dir = Path(tempfile.mkdtemp(prefix="eval_ppo_"))
    log.info("EnergyPlus scratch dir: %s", eplus_base_dir)

    new_rows: list[dict[str, Any]] = []
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
            row = eval_one_model(md, eplus_base_dir)
            new_rows.append(row)
            log.info(
                "  return=%.1f  mean_in_band=%.1f%%  worst_zone=%.1f%%",
                row["episode_return"],
                row["mean_pct_in_band"],
                row["worst_zone_pct_in_band"],
            )
        except Exception:
            log.exception(
                "FAILED: %s/%s/building_%d",
                md.building_type,
                md.task,
                md.building_index,
            )

    rows = existing_rows + new_rows

    # --- Post-processing: normalise against G36 baselines ---
    baselines = load_g36_baselines(args.g36_dir)
    for row in rows:
        key = (row["building_type"], row["task"], row["building_id"])
        g36_ret = baselines.get(key)
        if g36_ret is not None and g36_ret != 0.0:
            row["g36_return"] = round(g36_ret, 1)
            row["normalized_return"] = round(
                row["episode_return"] / g36_ret, 4
            )
        else:
            row["g36_return"] = ""
            row["normalized_return"] = ""

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d results (%d new) to %s", len(rows), len(new_rows), csv_path)

    # --- Summary ---
    expected = len(BUILDING_TYPES) * len(TASKS) * 8
    log.info(
        "\n%s\nSUMMARY: %d/%d evaluations completed (%d missing)\n%s",
        "=" * 70,
        len(rows),
        expected,
        expected - len(rows),
        "=" * 70,
    )

    if rows:
        df = pd.DataFrame(rows)
        for bt in df["building_type"].unique():
            bt_df = df[df["building_type"] == bt]
            log.info("  %s:", bt)
            for task in bt_df["task"].unique():
                task_df = bt_df[bt_df["task"] == task]
                mean_ret = task_df["episode_return"].mean()
                mean_band = task_df["mean_pct_in_band"].mean()
                norm_vals = pd.to_numeric(
                    task_df["normalized_return"], errors="coerce"
                ).dropna()
                norm_str = (
                    f"  norm_return={norm_vals.mean():.3f}"
                    if len(norm_vals) > 0
                    else ""
                )
                log.info(
                    "    %s: n=%d  mean_return=%.1f  mean_in_band=%.1f%%%s",
                    task,
                    len(task_df),
                    mean_ret,
                    mean_band,
                    norm_str,
                )


if __name__ == "__main__":
    main()
