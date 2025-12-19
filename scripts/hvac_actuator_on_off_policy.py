from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from algorithms.baselines import HVACActuatorOnOffPolicy
from algorithms.utils import make_env

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutPaths:
    out_dir: Path
    csv_path: Path
    npz_path: Path
    config_path: Path
    plot_temperature_path: Path
    plot_energy_path: Path
    plot_actuators_path: Path


def _make_rollout_paths(base_dir: Path) -> RolloutPaths:
    out_dir = base_dir / "hvac_actuator_on_off_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    return RolloutPaths(
        out_dir=out_dir,
        csv_path=out_dir / "rollout.csv",
        npz_path=out_dir / "rollout.npz",
        config_path=out_dir / "config_resolved.json",
        plot_temperature_path=out_dir / "temperature.png",
        plot_energy_path=out_dir / "energy.png",
        plot_actuators_path=out_dir / "actuators.png",
    )


def _require_env_metadata_list_str(env, key: str) -> list[str]:
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("env.metadata missing")
    raw = env.metadata.get(key)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"env.metadata['{key}'] must be a list[str]")
    return raw


def _plot_timeseries(
    *,
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    out_path: Path,
    title: str,
    ylabel: str,
    hlines: list[tuple[float, str]] | None = None,
) -> bool:
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


@hydra.main(version_base=None, config_path="../configs", config_name="hvac_actuator_on_off")
def main(cfg) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    run_dir = Path.cwd()
    paths = _make_rollout_paths(run_dir)

    # Save resolved config for reproducibility
    with paths.config_path.open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    eplus_output_dir = run_dir / "eplus_outputs"
    env = make_env(config=cfg, eplus_output_dir=str(eplus_output_dir))

    obs_names = _require_env_metadata_list_str(env, "observation_names")
    act_names = _require_env_metadata_list_str(env, "action_names")

    policy = HVACActuatorOnOffPolicy.from_env_metadata(
        env_metadata=env.metadata,  # type: ignore[arg-type]
        target_temp_c=float(cfg.policy.target_temp_c),
        deadband_c=float(cfg.policy.deadband_c),
        fan_mass_flow_kg_s=float(cfg.policy.fan_mass_flow_kg_s),
        terminal_mass_flow_kg_s=float(cfg.policy.terminal_mass_flow_kg_s),
        coil_speed_heat=float(cfg.policy.coil_speed_heat),
        coil_speed_cool=float(cfg.policy.coil_speed_cool),
        supplemental_stage_heat=float(cfg.policy.supplemental_stage_heat),
        availability_on=float(cfg.policy.availability_on),
        availability_off=float(cfg.policy.availability_off),
    )

    max_steps = int(getattr(cfg.env, "max_steps", 2000))
    n_episodes = int(getattr(cfg, "n_episodes", 1))
    target = float(cfg.policy.target_temp_c)
    deadband = float(cfg.policy.deadband_c)

    all_rows: list[dict[str, float]] = []

    for ep in range(n_episodes):
        obs, _info = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            action, _ = policy.predict(obs, deterministic=True)
            obs2, reward, terminated, truncated, _info = env.step(
                np.asarray(action, dtype=float)
            )

            row: dict[str, float] = {
                "episode": float(ep),
                "step": float(step),
                "reward": float(reward),
            }

            obs_arr = np.asarray(obs2, dtype=float).reshape(-1)
            for i, name in enumerate(obs_names):
                if i < len(obs_arr):
                    row[f"obs::{name}"] = float(obs_arr[i])

            act_arr = np.asarray(action, dtype=float).reshape(-1)
            for i, name in enumerate(act_names):
                if i < len(act_arr):
                    row[f"act::{name}"] = float(act_arr[i])

            all_rows.append(row)

            obs = obs2
            done = bool(terminated or truncated)
            step += 1

        logger.info(
            f"episode={ep} steps={step} last_mode={policy.last_mode} "
            f"saved_rows={len(all_rows)}"
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(paths.csv_path, index=False)

    # Also store dense arrays for fast loading
    obs_cols = [c for c in df.columns if c.startswith("obs::")]
    act_cols = [c for c in df.columns if c.startswith("act::")]
    np.savez_compressed(
        paths.npz_path,
        episode=df["episode"].to_numpy(dtype=np.int32),
        step=df["step"].to_numpy(dtype=np.int32),
        reward=df["reward"].to_numpy(dtype=float),
        obs_names=np.asarray(obs_cols, dtype=object),
        act_names=np.asarray(act_cols, dtype=object),
        obs=df[obs_cols].to_numpy(dtype=float) if obs_cols else np.zeros((len(df), 0)),
        act=df[act_cols].to_numpy(dtype=float) if act_cols else np.zeros((len(df), 0)),
    )

    # --- Graphs ---
    # Temperature plot: all zone temps + outdoor temp (if present)
    zone_temp_cols = [c for c in obs_cols if "zone air temperature" in c.lower()]
    outdoor_temp_cols = [c for c in obs_cols if c.lower().endswith("outdoor_temperature")]
    temp_cols = zone_temp_cols + outdoor_temp_cols
    _plot_timeseries(
        df=df,
        x="step",
        y_cols=temp_cols[:25],  # cap for readability
        out_path=paths.plot_temperature_path,
        title=f"Zone Temperatures (target={target:.1f}C deadband=±{deadband:.1f}C)",
        ylabel="Temperature [C]",
        hlines=[
            (target, "target"),
            (target - deadband, "target-deadband"),
            (target + deadband, "target+deadband"),
        ],
    )

    # Energy plot
    energy_cols = [c for c in obs_cols if c.lower().endswith("energy_electricity")] + [
        c for c in obs_cols if c.lower().endswith("energy_gas")
    ]
    _plot_timeseries(
        df=df,
        x="step",
        y_cols=energy_cols,
        out_path=paths.plot_energy_path,
        title="HVAC Energy (per-area, per-timestep)",
        ylabel="Wh / m2 / timestep",
    )

    # Actuators plot
    _plot_timeseries(
        df=df,
        x="step",
        y_cols=act_cols[:25],  # cap for readability
        out_path=paths.plot_actuators_path,
        title="HVAC Actuator Commands",
        ylabel="Actuator value (varies by actuator)",
    )

    logger.info(f"Saved raw rollout CSV: {paths.csv_path}")
    logger.info(f"Saved raw rollout NPZ: {paths.npz_path}")
    logger.info(f"Saved plots under: {paths.out_dir}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()


