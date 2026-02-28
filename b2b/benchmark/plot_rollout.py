"""Generic rollout plotting for any building type.

Reads the flat CSV produced by :func:`~b2b.benchmark.baseline_rollout.run_baseline_rollout`
and generates diagnostic PNG plots.  HVAC system type (VAV, Unitary, Baseboard)
is auto-detected from the ``act::`` column names.

Typical usage::

    from b2b.benchmark.plot_rollout import plot_rollout_csv
    plot_rollout_csv(Path("rollout.csv"), out_dir=Path("outputs/plots"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column classification helpers
# ---------------------------------------------------------------------------

def _zone_temp_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.lower().startswith("obs::zone air temperature")]


def _conditioned_zone_temp_cols(df: pd.DataFrame) -> list[str]:
    """Zone temps excluding obviously unconditioned spaces (attic, plenum, garage)."""
    skip = {"attic", "plenum", "garage", "unconditioned", "crawl"}
    return [
        c for c in _zone_temp_cols(df)
        if not any(s in c.lower() for s in skip)
    ]


def _outdoor_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in ("obs::outdoor_temperature",):
            return c
    return None


def _energy_cols(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "obs::energy_electricity":
            out["Electricity"] = c
        elif cl == "obs::energy_gas":
            out["Natural Gas"] = c
    return out


def _act_cols_by_type(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group ``act::`` columns by control type."""
    groups: dict[str, list[str]] = {
        "fan_flow": [],
        "sat_setpoint": [],
        "vav_flow_frac": [],
        "thermostat_htg": [],
        "thermostat_clg": [],
        "baseboard_avail": [],
        "other": [],
    }
    for c in df.columns:
        if not c.startswith("act::"):
            continue
        cl = c.lower()
        if "fan air mass flow rate" in cl:
            groups["fan_flow"].append(c)
        elif "temp" in cl or "setpoint" in cl:
            if "htg" in cl or "heating" in cl:
                groups["thermostat_htg"].append(c)
            elif "clg" in cl or "cooling" in cl:
                groups["thermostat_clg"].append(c)
            elif "vav" in cl and "flow" in cl:
                groups["vav_flow_frac"].append(c)
            else:
                groups["sat_setpoint"].append(c)
        elif "flow fraction" in cl or ("vav" in cl and "frac" in cl):
            groups["vav_flow_frac"].append(c)
        elif "availability" in cl or "baseboard" in cl:
            groups["baseboard_avail"].append(c)
        else:
            groups["other"].append(c)
    return {k: v for k, v in groups.items() if v}


def _short_label(col: str) -> str:
    """Derive a short legend label from a full column name."""
    raw = col
    if raw.startswith("obs::"):
        raw = raw[5:]
    elif raw.startswith("act::"):
        raw = raw[5:]
    parts = raw.split("::")
    if len(parts) >= 3:
        return parts[2]
    if len(parts) >= 1:
        return parts[-1]
    return raw


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_temperatures(
    df: pd.DataFrame,
    out_path: Path,
    *,
    target_c: float | None = 21.0,
) -> None:
    import matplotlib.pyplot as plt

    zone_cols = _conditioned_zone_temp_cols(df)
    if not zone_cols:
        zone_cols = _zone_temp_cols(df)
    if not zone_cols:
        logger.warning("No zone temperature observations found — skipping temperature plot")
        return

    x = df["global_step"] if "global_step" in df.columns else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(14, 6))
    for c in zone_cols:
        ax.plot(x, df[c], label=_short_label(c), alpha=0.8, linewidth=0.8)

    outdoor = _outdoor_temp_col(df)
    if outdoor is not None:
        ax.plot(x, df[outdoor], label="Outdoor", color="black", linestyle=":",
                alpha=0.7, linewidth=1.2)

    if target_c is not None:
        ax.axhline(target_c, color="gray", linestyle="--", alpha=0.5,
                    label=f"Target {target_c}°C")

    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Zone Air Temperatures")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_actuators(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    groups = _act_cols_by_type(df)
    if not groups:
        logger.warning("No actuator columns found — skipping actuator plots")
        return

    x = df["global_step"] if "global_step" in df.columns else np.arange(len(df))

    plot_specs: list[tuple[str, str, str, str | None]] = [
        ("fan_flow", "Fan Air Mass Flow Rate", "Flow (kg/s)", None),
        ("sat_setpoint", "Supply Air Temperature Setpoint", "Temperature (°C)", None),
        ("vav_flow_frac", "VAV Flow Fraction (Damper Position)", "Fraction (0–1)", None),
        ("thermostat_htg", "Heating Setpoints", "Temperature (°C)", None),
        ("thermostat_clg", "Cooling Setpoints", "Temperature (°C)", None),
        ("baseboard_avail", "Baseboard Availability", "On/Off", None),
        ("other", "Other Actuators", "", None),
    ]

    any_plotted = False
    for group_key, title, ylabel, _ref in plot_specs:
        cols = groups.get(group_key, [])
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        for c in cols:
            ax.plot(x, df[c], label=_short_label(c), alpha=0.8, linewidth=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"actuators_{group_key}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        any_plotted = True

    if not any_plotted:
        return

    # Combined actuator overview (all groups on subplots)
    non_empty = [(k, v) for k, v in groups.items() if v]
    if len(non_empty) <= 1:
        return
    n_panels = min(len(non_empty), 4)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    for ax_i, (key, cols) in zip(axes, non_empty[:n_panels]):
        for c in cols:
            ax_i.plot(x, df[c], label=_short_label(c), alpha=0.8, linewidth=0.7)
        ax_i.set_ylabel(key.replace("_", " ").title())
        ax_i.legend(loc="upper right", fontsize=7, ncol=2)
        ax_i.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle("Actuator Overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "actuators_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_energy(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    energy = _energy_cols(df)
    if not energy:
        logger.warning("No energy observations found — skipping energy plot")
        return

    x = df["global_step"] if "global_step" in df.columns else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 5))
    total = np.zeros(len(df))
    for label, col in energy.items():
        vals = df[col].values.astype(float)
        ax.plot(x, vals, label=label, alpha=0.8)
        total += vals
    if len(energy) > 1:
        ax.plot(x, total, label="Total", linestyle="--", alpha=0.7, color="black")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (Wh/m²)")
    ax.set_title("Energy Consumption")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_reward(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if "reward" not in df.columns:
        return

    x = df["global_step"] if "global_step" in df.columns else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, df["reward"], alpha=0.5, linewidth=0.5, color="steelblue")
    window = min(96, len(df) // 4) or 1
    ax.plot(x, df["reward"].rolling(window, min_periods=1).mean(),
            color="navy", linewidth=1.5, label=f"Rolling mean ({window} steps)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def plot_rollout_df(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    target_c: float | None = 21.0,
) -> list[Path]:
    """Generate all diagnostic plots from a rollout DataFrame.

    Args:
        df: DataFrame with ``obs::*``, ``act::*``, ``reward`` columns
            (as produced by :func:`run_baseline_rollout`).
        out_dir: Directory to write PNG files into.
        target_c: Reference temperature line on the temperature plot.

    Returns:
        List of paths to the generated plot files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    temp_path = out_dir / "temperature.png"
    _plot_temperatures(df, temp_path, target_c=target_c)
    if temp_path.exists():
        created.append(temp_path)

    _plot_actuators(df, out_dir)
    for p in sorted(out_dir.glob("actuators_*.png")):
        created.append(p)

    energy_path = out_dir / "energy.png"
    _plot_energy(df, energy_path)
    if energy_path.exists():
        created.append(energy_path)

    reward_path = out_dir / "reward.png"
    _plot_reward(df, reward_path)
    if reward_path.exists():
        created.append(reward_path)

    return created


def plot_rollout_csv(
    csv_path: Path,
    out_dir: Path | None = None,
    *,
    target_c: float | None = 21.0,
) -> list[Path]:
    """Generate diagnostic plots from a rollout CSV file.

    Args:
        csv_path: Path to the ``rollout.csv`` produced by the baseline runner.
        out_dir: Directory to write PNG files.  Defaults to ``csv_path.parent``.
        target_c: Reference temperature line on the temperature plot.

    Returns:
        List of paths to the generated plot files.
    """
    if out_dir is None:
        out_dir = csv_path.parent
    df = pd.read_csv(csv_path)
    return plot_rollout_df(df, out_dir, target_c=target_c)
