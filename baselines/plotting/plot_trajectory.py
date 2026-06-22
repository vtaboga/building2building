"""Per-episode trajectory plots: zone temperatures, setpoints, outdoor air, and actuators.

Usage (standalone)::

    python -m baselines.plotting.plot_trajectory \\
        --trajectory trajectories/OfficeSmall_1234_task1.npz \\
        --output figures/trajectory_OfficeSmall_1234_task1

Or call :func:`plot_trajectory` programmatically from evaluation scripts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from baselines.plotting.common import apply_paper_style, save_figure

FIGSIZE_TRAJECTORY = (14.0, 8.0)


@dataclass
class TrajectoryData:
    """All data needed to plot one episode trajectory.

    Stored in / loaded from a ``.npz`` file with matching array keys
    plus JSON-safe metadata in ``meta``.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    observation_names: list[str]
    action_names: list[str]

    zone_temp_indices: list[int] = field(default_factory=list)
    zone_temp_labels: list[str] = field(default_factory=list)

    setpoint_indices: list[int] = field(default_factory=list)
    setpoint_labels: list[str] = field(default_factory=list)

    outdoor_temp_index: int | None = None

    building_type: str = ""
    building_id: str = ""
    task: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        meta = json.dumps(
            {
                "observation_names": self.observation_names,
                "action_names": self.action_names,
                "zone_temp_indices": self.zone_temp_indices,
                "zone_temp_labels": self.zone_temp_labels,
                "setpoint_indices": self.setpoint_indices,
                "setpoint_labels": self.setpoint_labels,
                "outdoor_temp_index": self.outdoor_temp_index,
                "building_type": self.building_type,
                "building_id": self.building_id,
                "task": self.task,
            }
        )
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            meta=np.array(meta),
        )

    @classmethod
    def load(cls, path: Path) -> "TrajectoryData":
        import json

        data = np.load(path, allow_pickle=False)
        meta: dict[str, Any] = json.loads(str(data["meta"]))
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            observation_names=meta["observation_names"],
            action_names=meta["action_names"],
            zone_temp_indices=meta.get("zone_temp_indices", []),
            zone_temp_labels=meta.get("zone_temp_labels", []),
            setpoint_indices=meta.get("setpoint_indices", []),
            setpoint_labels=meta.get("setpoint_labels", []),
            outdoor_temp_index=meta.get("outdoor_temp_index"),
            building_type=meta.get("building_type", ""),
            building_id=meta.get("building_id", ""),
            task=meta.get("task", ""),
        )


def extract_trajectory_data(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    metadata: dict[str, Any],
    *,
    building_type: str = "",
    building_id: str = "",
    task: str = "",
) -> TrajectoryData:
    """Build a :class:`TrajectoryData` from an episode result and env metadata."""
    obs_names: list[str] = metadata.get("observation_names", [])
    act_names: list[str] = metadata.get("action_names", [])

    zone_temp_prefix = "zone air temperature"
    zone_temp_indices: list[int] = []
    zone_temp_labels: list[str] = []
    for i, name in enumerate(obs_names):
        if name.strip().lower().startswith(zone_temp_prefix):
            zone_temp_indices.append(i)
            zone_part = name.strip()[len("ZONE AIR TEMPERATURE") :].strip()
            zone_temp_labels.append(zone_part if zone_part else f"zone_{i}")

    setpoint_prefix = "target_temperature"
    setpoint_indices: list[int] = []
    setpoint_labels: list[str] = []
    for i, name in enumerate(obs_names):
        if name.strip().lower().startswith(setpoint_prefix):
            zone_part = name.strip()[len("target_temperature") :].strip()
            setpoint_indices.append(i)
            setpoint_labels.append(zone_part if zone_part else f"setpoint_{i}")

    outdoor_temp_index: int | None = None
    for i, name in enumerate(obs_names):
        if name.strip().lower() == "outdoor_temperature":
            outdoor_temp_index = i
            break

    return TrajectoryData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        observation_names=obs_names,
        action_names=act_names,
        zone_temp_indices=zone_temp_indices,
        zone_temp_labels=zone_temp_labels,
        setpoint_indices=setpoint_indices,
        setpoint_labels=setpoint_labels,
        outdoor_temp_index=outdoor_temp_index,
        building_type=building_type,
        building_id=building_id,
        task=task,
    )


def _short_actuator_label(action_name: str) -> str:
    """Shorten ``component_type::control_type::component_name`` for legends."""
    parts = action_name.split("::")
    if len(parts) >= 2:
        return f"{parts[1].strip()}"
    return action_name


def plot_trajectory(
    traj: TrajectoryData,
    *,
    output_path: Path | None = None,
    formats: Sequence[str] = ("pdf", "png"),
    max_actuators: int = 12,
) -> plt.Figure:
    """Plot zone temperatures, setpoints, outdoor temp, and actuator values.

    Returns the matplotlib Figure (caller may show or save it).
    """
    apply_paper_style()

    has_temps = bool(traj.zone_temp_indices)
    has_actuators = traj.actions.ndim == 2 and traj.actions.shape[1] > 0

    n_panels = 0
    if has_temps:
        n_panels += 1
    if has_actuators:
        n_panels += 1
    n_panels = max(n_panels, 1)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(FIGSIZE_TRAJECTORY[0], 4.0 * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes_flat: list[plt.Axes] = [ax for row in axes for ax in row]

    title_parts = []
    if traj.building_type:
        title_parts.append(traj.building_type)
    if traj.building_id:
        title_parts.append(traj.building_id)
    if traj.task:
        title_parts.append(traj.task)
    if title_parts:
        fig.suptitle(" / ".join(title_parts), fontsize=11)

    panel_idx = 0

    # --- Temperature panel ---
    if has_temps:
        ax_temp = axes_flat[panel_idx]
        panel_idx += 1

        T = traj.observations.shape[0]
        timesteps = np.arange(T)
        hours = timesteps / 6.0  # 10-min default timestep → hours

        cmap = plt.get_cmap("tab10")

        for k, (obs_idx, label) in enumerate(
            zip(traj.zone_temp_indices, traj.zone_temp_labels)
        ):
            color = cmap(k % 10)
            ax_temp.plot(
                hours,
                traj.observations[:, obs_idx],
                label=f"Zone: {label}",
                color=color,
                linewidth=0.8,
            )

        for k, (obs_idx, label) in enumerate(
            zip(traj.setpoint_indices, traj.setpoint_labels)
        ):
            color = cmap(k % 10)
            ax_temp.plot(
                hours,
                traj.observations[:, obs_idx],
                label=f"Setpoint: {label}",
                color=color,
                linewidth=0.8,
                linestyle="--",
            )

        if traj.outdoor_temp_index is not None:
            ax_temp.plot(
                hours,
                traj.observations[:, traj.outdoor_temp_index],
                label="Outdoor",
                color="grey",
                linewidth=0.8,
                linestyle=":",
            )

        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            ncol=1,
        )

    # --- Actuator panel ---
    if has_actuators:
        ax_act = axes_flat[panel_idx]
        panel_idx += 1

        n_act = traj.actions.shape[1]
        T_act = traj.actions.shape[0]
        act_hours = np.arange(T_act) / 6.0

        act_cmap = plt.get_cmap("Set2")
        n_show = min(n_act, max_actuators)

        for j in range(n_show):
            label = (
                _short_actuator_label(traj.action_names[j])
                if j < len(traj.action_names)
                else f"act_{j}"
            )
            ax_act.plot(
                act_hours,
                traj.actions[:, j],
                label=label,
                color=act_cmap(j % 8),
                linewidth=0.7,
                alpha=0.85,
            )

        if n_act > max_actuators:
            ax_act.text(
                0.99,
                0.01,
                f"({n_act - max_actuators} more actuators not shown)",
                transform=ax_act.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                fontstyle="italic",
            )

        ax_act.set_ylabel("Actuator value")
        ax_act.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            ncol=1,
        )

    axes_flat[-1].set_xlabel("Time (hours)")

    fig.tight_layout(rect=[0, 0, 0.82, 0.96])

    if output_path is not None:
        save_figure(fig, output_path, formats=formats)

    return fig


# ---- CLI entry point ----


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved trajectory (.npz).")
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="Path to a .npz trajectory file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory_plot",
        help="Output path stem (without extension).",
    )
    args = parser.parse_args()

    traj = TrajectoryData.load(Path(args.trajectory))
    fig = plot_trajectory(traj, output_path=Path(args.output))
    plt.close(fig)


if __name__ == "__main__":
    main()
