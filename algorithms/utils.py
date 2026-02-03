from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from pathlib import Path
from typing import Iterable

import pandas as pd
from building2building.simulator import create_simulator
from building2building.sources import hydroquebec
from building2building.utils import (
    hydroquebec_building_id_from_split_index,
    hydroquebec_filenames_for_building_id,
)
from omegaconf import OmegaConf
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from algorithms.wandb_utils import wandb_log_xy_series

logger = logging.getLogger(__name__)


def make_env(config, eplus_output_dir: str):
    # EnergyPlus needs a unique output dir for each run
    eplus_output_dir = Path(eplus_output_dir) / str(uuid.uuid4())
    eplus_output_dir.mkdir(parents=True, exist_ok=True)
    # Fetch exactly one BuildingConfig and build a single simulator env
    try:
        # Optional: deterministic building selection from train/test row-id lists.
        #
        # Config shape (recommended):
        #   bldg:
        #     selection:
        #       enabled: true
        #       split: train|test
        #       index: 0
        #
        # This injects an `idf_filename` filter so `hydroquebec.search_configs(...)`
        # selects exactly that building.
        cfg_any = config
        if not isinstance(cfg_any, dict):
            try:
                cfg_any = OmegaConf.to_container(cfg_any, resolve=True)  # type: ignore[assignment]
            except Exception:
                cfg_any = config

        if isinstance(cfg_any, dict):
            bldg_section = cfg_any.get("bldg")
            if isinstance(bldg_section, dict):
                sel = bldg_section.get("selection")
                if isinstance(sel, dict) and bool(sel.get("enabled", False)):
                    split = str(sel.get("split", "train")).strip().lower()
                    if split not in ("train", "test"):
                        raise ValueError(f"bldg.selection.split must be 'train' or 'test', got {split!r}")
                    split_idx_raw = sel.get("index", 0)
                    if not isinstance(split_idx_raw, int):
                        raise TypeError(
                            f"bldg.selection.index must be int, got {type(split_idx_raw).__name__}"
                        )
                    building_id = hydroquebec_building_id_from_split_index(
                        split=split, split_index=split_idx_raw
                    )
                    idf_filename, schedule_filename = hydroquebec_filenames_for_building_id(building_id)

                    # Override any existing building query: pick exactly this building.
                    bldg_section["bldg"] = {
                        "idf_filename": idf_filename,
                        "schedule_filename": schedule_filename,
                    }
                    cfg_any["bldg"] = bldg_section
                    config = cfg_any

        # Let the pipeline copy discovery `eplusout.err` into this run folder.
        prev = os.environ.get("B2B_PIPELINE_DEBUG_DIR")
        os.environ["B2B_PIPELINE_DEBUG_DIR"] = str(eplus_output_dir)
        configs = hydroquebec.search_configs(
            config=config, n=1, eplus_output_dir=Path(eplus_output_dir)
        )
        if not configs:
            raise RuntimeError(
                "No building configurations found for the provided config "
                "(see pipeline_errors.jsonl in the EnergyPlus output dir if present)."
            )
        env_config = configs[0]
        env = create_simulator(env_config)
        return env
    except Exception as e:
        # Persist a structured error record to make batch runs debuggable.
        err_path = Path(eplus_output_dir) / "env_creation_error.json"
        record = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            with err_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception:
            logger.exception("Failed to write env creation error to %s", err_path)
        raise
    finally:
        # Restore previous value to avoid leaking run-specific state.
        if "prev" in locals():
            if prev is None:
                os.environ.pop("B2B_PIPELINE_DEBUG_DIR", None)
            else:
                os.environ["B2B_PIPELINE_DEBUG_DIR"] = prev


def make_dummy_vec_env(config, eplus_output_dir: str, seed: int | None = None, wrapper_fn=None):
    """Create a single-env DummyVecEnv with an optional wrapper_fn applied."""
    def _thunk():
        # seed is accepted for API compatibility but not used here
        env = make_env(config=config, eplus_output_dir=eplus_output_dir)
        env = Monitor(env)
        if wrapper_fn is not None:
            env = wrapper_fn(env)
        return env
    return DummyVecEnv([_thunk])


def plot_timeseries(
    *,
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    out_path: Path,
    title: str,
    ylabel: str,
    hlines: list[tuple[float, str]] | None = None,
) -> bool:
    """
    Best-effort matplotlib timeseries plot.

    Returns False when matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning(f"matplotlib not available; skipping plot {out_path.name}: {e}")
        return False

    plt.figure(figsize=(12, 5))
    for c in y_cols:
        if c in df.columns:
            plt.plot(df[x].to_numpy(), df[c].to_numpy(), label=c, linewidth=1.2)
    if hlines:
        for y, lbl in hlines:
            plt.axhline(y=y, linestyle="--", linewidth=1.0, label=lbl)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    if len(y_cols) <= 12:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _log_line_series(df: pd.DataFrame):
    """Log one W&B line chart per column under 'Test Graphs/' averaged by 4.

    Assumes dataset length is a fixed multiple of 4 and data are numeric.
    """
    if df is None or df.empty or df.shape[1] == 0:
        return

    n = len(df)

    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").to_numpy()
        # Average each 4 consecutive elements (n is multiple of 4)
        y = series.reshape(-1, 4).mean(axis=1)
        xs = list(range(0, n//4))
        wandb_log_xy_series(
            x=xs,
            y=[float(v) for v in y],
            key=f"Test Graphs/{col}",
            title=str(col),
            x_name="timestep",
            y_name="value",
        )


def log_test_graphs_wandb(test_csv: str | Path):
    """Log 4 quick-look graphs to Weights & Biases from a test CSV.

    Graphs:
      1) Zone air temperatures + outdoor temperature
      2) Actions
      3) Energy gas and electricity
      4) Reward

    X-axis is simulation timestep (row index).
    """

    csv_path = Path(test_csv)
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # Case-insensitive lookup map
    lower_map = {c.lower(): c for c in cols}

    # Outdoor and zone air temperatures
    temperature_cols = [c for c in cols if c.lower().startswith("zone air temperature")]
    outdoor_col =  lower_map.get("outdoor_temperature")
    if outdoor_col:
        temperature_cols.append(outdoor_col)
    if temperature_cols:
        _log_line_series(df[temperature_cols])

    # Actions
    action_cols = [c for c in cols if c.lower().startswith("action")]
    if action_cols:
        _log_line_series(df[action_cols])

    # Energy consumption
    energy_candidates = [lower_map.get("energy_electricity"), lower_map.get("energy_gas")]
    energy_cols = [c for c in energy_candidates if c]
    
    if energy_cols:
        _log_line_series(df[energy_cols])

    # Reward
    reward_col = lower_map.get("reward")
    if reward_col:
        _log_line_series(df[[reward_col]])


def log_test_dir_graphs_wandb(test_dir: str | Path):
    """Convenience: log graphs for each policy_episode_*.csv in a directory."""

    tdir = Path(test_dir)
    files = sorted(tdir.glob("policy_episode_*.csv"))
    for csv_path in files:
        log_test_graphs_wandb(csv_path)


