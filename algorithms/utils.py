from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Iterable

import pandas as pd
from building2building.simulator import create_simulator
from building2building.sources import hydroquebec
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from algorithms.wandb_utils import wandb_log_xy_series

logger = logging.getLogger(__name__)


def make_env(config, eplus_output_dir: str):
    # EnergyPlus needs a unique output dir for each run
    eplus_output_dir = Path(eplus_output_dir) / str(uuid.uuid4())
    # Fetch exactly one BuildingConfig and build a single simulator env
    configs = hydroquebec.search_configs(config=config, n=1, eplus_output_dir=Path(eplus_output_dir))
    if not configs:
        raise RuntimeError("No building configurations found for the provided config.")
    env_config = configs[0]
    env = create_simulator(env_config)
    return env


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


