from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def plot_actuators_dual_axis(
    *,
    df: pd.DataFrame,
    x: str,
    act_cols: list[str],
    out_path: Path,
) -> bool:
    """
    Plot HVAC actuator commands with improved scaling:

    - Removes availability commands (AirLoopHVAC availability status)
    - Plots temperature setpoints on the left y-axis
    - Plots airflow (fan mass flow) on the right y-axis
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning(f"matplotlib not available; skipping plot {out_path.name}: {e}")
        return False

    cols = [c for c in act_cols if c in df.columns]
    if not cols:
        return False

    def _is_availability(col: str) -> bool:
        s = col.lower()
        return ("availability status" in s) and ("airloophvac" in s)

    def _is_airflow(col: str) -> bool:
        s = col.lower()
        return ("fan air mass flow rate" in s) and ("fan::" in s)

    def _is_setpoint(col: str) -> bool:
        s = col.lower()
        # Prefer explicit B2B node temp schedules, but also accept direct node setpoints.
        if "b2b node temp sp" in s and "schedule value" in s:
            return True
        if "system node setpoint" in s and "temperature setpoint" in s:
            return True
        return False

    cols = [c for c in cols if not _is_availability(c)]
    airflow_cols = [c for c in cols if _is_airflow(c)]
    setpoint_cols = [c for c in cols if _is_setpoint(c)]

    # If heuristics fail, fall back to plotting all non-availability actions on one axis.
    if not airflow_cols and not setpoint_cols:
        setpoint_cols = cols

    fig, ax_left = plt.subplots(figsize=(12, 5))
    xs = df[x].to_numpy()

    # Left axis: setpoints
    left_lines = []
    left_labels = []
    for c in setpoint_cols:
        line = ax_left.plot(xs, df[c].to_numpy(), label=c, linewidth=1.2)
        if line:
            left_lines.append(line[0])
            left_labels.append(c)
    ax_left.set_ylabel("Temperature setpoints (varies by actuator)")

    # Right axis: airflow
    ax_right = ax_left.twinx()
    right_lines = []
    right_labels = []
    for c in airflow_cols:
        line = ax_right.plot(
            xs, df[c].to_numpy(), label=c, linewidth=1.2, linestyle="--"
        )
        if line:
            right_lines.append(line[0])
            right_labels.append(c)
    ax_right.set_ylabel("Airflow (fan mass flow) (varies by actuator)")

    ax_left.set_title("HVAC Actuator Commands")
    ax_left.set_xlabel(x)
    ax_left.grid(True, alpha=0.2)

    lines = left_lines + right_lines
    labels = left_labels + right_labels
    if len(labels) <= 12 and labels:
        ax_left.legend(lines, labels, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True

