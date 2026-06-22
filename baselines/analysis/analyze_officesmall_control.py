#!/usr/bin/env python3
"""Diagnostic rollouts for the tuned OfficeSmall unitary-HVAC controller.

Generates full-year rollouts on a user-specified list of OfficeSmall
buildings, loads the matching tuned controller config (by climate zone),
and saves both the raw trajectory and a quantitative analysis:

    (1) Per-zone temperature distribution vs. the 21 C setpoint (percentiles,
        deadband fractions).
    (2) Actuator (fan airflow and supply-air-temperature setpoint) statistics.
    (3) Per-step decomposition of the deadband reward into its temperature
        and energy terms so the relative order of magnitude is visible.

Usage::

    python -m baselines.analysis.analyze_officesmall_control \\
        --output-dir analysis/officesmall_tuned \\
        --buildings OfficeSmall-5167:1 OfficeSmall-5236:4 OfficeSmall-5022:8

The format ``<building_id>:<climate_zone>`` tells the script which tuned
YAML to load (``unitary_hvac_officesmall_cz{cz}.yaml``).  Because
``_get_climate_zone`` in ``run_reactive_control`` currently fails on the
hashed weather filenames, the climate zone is passed explicitly rather
than auto-detected.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

import building2building as b2b
from baselines.controllers.unitary_hvac import UnitaryHvacConfig, UnitaryHvacPolicy

logger = logging.getLogger(__name__)

TUNED_CONFIGS_DIR = (
    Path(__file__).resolve().parent.parent / "configs" / "tuned_controllers"
)

TARGET_C = 21.0
DEADBAND_C = 1.0
ENERGY_WEIGHT = 0.01


@dataclass
class ZoneStep:
    """One step of ground-truth observables per controlled zone."""

    temps_by_zone: dict[str, float]
    target: float
    outdoor_c: float
    energy_electricity: float
    energy_gas: float
    reward: float


def _load_tuned_config(cz: int) -> UnitaryHvacConfig:
    p = TUNED_CONFIGS_DIR / f"unitary_hvac_officesmall_cz{cz}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Tuned config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    raw.pop("type", None)
    raw.pop("target_schedule", None)
    return UnitaryHvacConfig(
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in raw.items()}
    )


def _deadband_reward_breakdown(
    zone_temps: dict[str, float],
    target: float,
    dT: float,
    energy_electricity: float,
    energy_gas: float,
    energy_weight: float,
) -> tuple[float, float, float]:
    """Recompute ``task1`` reward and return (temp_term, energy_term, total).

    Mirrors :func:`building2building.simulator.rewards.deadband_reward_function`
    so we can attribute per-step reward to each component.

    Returns:
        temp_term: ``mean_z f(T_z - target)`` where ``f(x) = x^2`` inside the
            deadband and ``f(x) = |x|`` outside.  Always non-negative; the
            reward subtracts this.
        energy_term: ``energy_weight * (electricity + gas)`` in reward units.
        total: ``-(temp_term + energy_term)`` (matches env reward).
    """
    if not zone_temps:
        return 0.0, 0.0, 0.0
    acc = 0.0
    for _zone, t in zone_temps.items():
        dev = abs(t - target)
        if dev <= dT:
            acc += (t - target) ** 2
        else:
            acc += dev
    temp_term = acc / len(zone_temps)
    energy_term = energy_weight * (energy_electricity + energy_gas)
    return temp_term, energy_term, -(temp_term + energy_term)


def run_rollout(
    building_id: str,
    climate_zone: int,
    output_dir: Path,
) -> dict[str, Any]:
    cfg = _load_tuned_config(climate_zone)
    logger.info(
        "Building %s (cz=%d) — running full-year rollout with tuned config",
        building_id,
        climate_zone,
    )

    eplus_dir = output_dir / "eplus" / building_id
    eplus_dir.mkdir(parents=True, exist_ok=True)
    env = b2b.make_env(
        "OfficeSmall",
        building_id=building_id,
        task="task1",
        run_period="full_year",
        eplus_output_dir=eplus_dir,
    )
    controlled_zones: list[str] = list(env.metadata["controlled_zones"])
    action_names: list[str] = list(env.metadata["action_names"])

    try:
        policy = UnitaryHvacPolicy(cfg)
        policy.bind_env(env)

        obs_vec, info = env.reset()
        raw = info["raw_observation"]

        temps_hist: list[dict[str, float]] = [
            {z: float(raw["temperature"][z]) for z in controlled_zones}
        ]
        outdoor_hist: list[float] = [float(raw["outdoor"]["temperature"])]
        energy_e_hist: list[float] = [float(raw["energy"]["electricity"])]
        energy_g_hist: list[float] = [float(raw["energy"]["natural_gas"])]
        target_hist: list[float] = [TARGET_C]
        action_hist: list[np.ndarray] = []
        reward_hist: list[float] = []
        temp_term_hist: list[float] = []
        energy_term_hist: list[float] = []

        done = False
        while not done:
            action, _ = policy.predict(obs_vec, deterministic=True)
            obs_vec, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            raw = info.get("raw_observation")
            if raw is None:
                continue
            zt = {z: float(raw["temperature"][z]) for z in controlled_zones}
            energy_e = float(raw["energy"]["electricity"])
            energy_g = float(raw["energy"]["natural_gas"])
            t_term, e_term, _ = _deadband_reward_breakdown(
                zt, TARGET_C, DEADBAND_C, energy_e, energy_g, ENERGY_WEIGHT
            )

            temps_hist.append(zt)
            outdoor_hist.append(float(raw["outdoor"]["temperature"]))
            energy_e_hist.append(energy_e)
            energy_g_hist.append(energy_g)
            target_hist.append(TARGET_C)
            action_hist.append(np.asarray(action, dtype=np.float32))
            reward_hist.append(float(reward))
            temp_term_hist.append(t_term)
            energy_term_hist.append(e_term)
    finally:
        env.close()

    zones_arr = np.array(
        [[step[z] for z in controlled_zones] for step in temps_hist],
        dtype=np.float32,
    )
    actions_arr = (
        np.stack(action_hist).astype(np.float32)
        if action_hist
        else np.zeros((0, len(action_names)), dtype=np.float32)
    )
    summary = {
        "building_id": building_id,
        "climate_zone": climate_zone,
        "controlled_zones": controlled_zones,
        "action_names": action_names,
        "target_c": TARGET_C,
        "deadband_c": DEADBAND_C,
        "energy_weight": ENERGY_WEIGHT,
        "episode_steps": int(actions_arr.shape[0]),
        "total_reward": float(sum(reward_hist)),
        "sum_temp_term": float(sum(temp_term_hist)),
        "sum_energy_term": float(sum(energy_term_hist)),
    }
    traj_path = output_dir / f"{building_id}_cz{climate_zone}.npz"
    np.savez_compressed(
        traj_path,
        zone_temperatures=zones_arr,
        outdoor_temperature=np.asarray(outdoor_hist, dtype=np.float32),
        energy_electricity=np.asarray(energy_e_hist, dtype=np.float32),
        energy_gas=np.asarray(energy_g_hist, dtype=np.float32),
        actions=actions_arr,
        rewards=np.asarray(reward_hist, dtype=np.float32),
        temp_term=np.asarray(temp_term_hist, dtype=np.float32),
        energy_term=np.asarray(energy_term_hist, dtype=np.float32),
        meta=np.array(json.dumps(summary)),
    )
    logger.info("Saved trajectory → %s", traj_path)
    return summary


def analyze_one(
    npz_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute quantitative stats + plots for a single building rollout."""
    data = np.load(npz_path, allow_pickle=False)
    meta: dict[str, Any] = json.loads(str(data["meta"]))
    zones: list[str] = meta["controlled_zones"]
    action_names: list[str] = meta["action_names"]
    target = float(meta["target_c"])
    dT = float(meta["deadband_c"])

    temps: np.ndarray = data["zone_temperatures"]  # (T, n_zones)
    outdoor: np.ndarray = data["outdoor_temperature"]
    energy_e: np.ndarray = data["energy_electricity"]
    energy_g: np.ndarray = data["energy_gas"]
    actions: np.ndarray = data["actions"]
    rewards: np.ndarray = data["rewards"]
    temp_term: np.ndarray = data["temp_term"]
    energy_term: np.ndarray = data["energy_term"]

    dev = temps - target

    per_zone = []
    for i, z in enumerate(zones):
        d = dev[:, i]
        pct = np.percentile(d, [1, 5, 25, 50, 75, 95, 99])
        per_zone.append(
            {
                "zone": z,
                "mean_dev_c": float(d.mean()),
                "std_dev_c": float(d.std()),
                "pct_1": float(pct[0]),
                "pct_5": float(pct[1]),
                "pct_25": float(pct[2]),
                "pct_50": float(pct[3]),
                "pct_75": float(pct[4]),
                "pct_95": float(pct[5]),
                "pct_99": float(pct[6]),
                "frac_in_deadband": float(np.mean(np.abs(d) <= dT)),
                "frac_above_deadband": float(np.mean(d > dT)),
                "frac_below_deadband": float(np.mean(-d > dT)),
                "max_over_c": float(d.max()),
                "max_under_c": float((-d).max()),
            }
        )

    all_dev = dev.reshape(-1)
    temp_stats = {
        "per_zone": per_zone,
        "aggregate": {
            "mean_dev_c": float(all_dev.mean()),
            "std_dev_c": float(all_dev.std()),
            "frac_in_deadband": float(np.mean(np.abs(all_dev) <= dT)),
            "frac_above_deadband": float(np.mean(all_dev > dT)),
            "frac_below_deadband": float(np.mean(-all_dev > dT)),
            "max_over_c": float(all_dev.max()),
            "max_under_c": float((-all_dev).max()),
        },
    }

    fan_idx = [i for i, n in enumerate(action_names) if "Fan Air Mass Flow Rate" in n]
    sat_idx = [i for i, n in enumerate(action_names) if "Schedule Value" in n]
    action_stats = {
        "n_steps": int(actions.shape[0]),
        "fan_actuators": [
            {
                "name": action_names[i],
                "mean": float(actions[:, i].mean()),
                "min": float(actions[:, i].min()),
                "max": float(actions[:, i].max()),
                "std": float(actions[:, i].std()),
                "frac_at_min": float(
                    np.mean(actions[:, i] <= actions[:, i].min() + 1e-6)
                ),
            }
            for i in fan_idx
        ],
        "sat_actuators": [
            {
                "name": action_names[i],
                "mean": float(actions[:, i].mean()),
                "min": float(actions[:, i].min()),
                "max": float(actions[:, i].max()),
                "std": float(actions[:, i].std()),
            }
            for i in sat_idx
        ],
    }

    n = len(rewards)
    reward_stats = {
        "total_reward": float(rewards.sum()),
        "mean_reward_per_step": float(rewards.mean()),
        "sum_temp_term": float(temp_term.sum()),
        "sum_energy_term": float(energy_term.sum()),
        "mean_temp_term_per_step": float(temp_term.mean()),
        "mean_energy_term_per_step": float(energy_term.mean()),
        "share_of_penalty_from_temp": (
            float(temp_term.sum() / (temp_term.sum() + energy_term.sum()))
            if (temp_term.sum() + energy_term.sum()) > 0
            else float("nan")
        ),
        "share_of_penalty_from_energy": (
            float(energy_term.sum() / (temp_term.sum() + energy_term.sum()))
            if (temp_term.sum() + energy_term.sum()) > 0
            else float("nan")
        ),
        "outdoor_temperature_c": {
            "mean": float(outdoor.mean()),
            "min": float(outdoor.min()),
            "max": float(outdoor.max()),
        },
    }

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=False)
    stem = npz_path.stem

    ax = axes[0]
    hours = np.arange(temps.shape[0]) * (60.0 / 12.0) / 60.0  # 12 ts/hour
    for i, z in enumerate(zones):
        ax.plot(hours, temps[:, i], lw=0.4, alpha=0.7, label=z)
    ax.axhline(target, color="k", lw=1, ls="--", label=f"setpoint={target}°C")
    ax.axhline(target + dT, color="grey", lw=0.6, ls=":", label=f"±{dT}°C deadband")
    ax.axhline(target - dT, color="grey", lw=0.6, ls=":")
    ax.plot(hours, outdoor, lw=0.3, color="tab:brown", alpha=0.5, label="outdoor")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Hour of year")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title(
        f"{meta['building_id']} (cz={meta['climate_zone']}) — zone temperatures"
    )

    ax = axes[1]
    bins = np.linspace(-12, 12, 97)
    for i, z in enumerate(zones):
        ax.hist(
            dev[:, i],
            bins=bins,
            alpha=0.35,
            label=z,
            density=True,
            histtype="stepfilled",
        )
    ax.axvspan(-dT, dT, color="green", alpha=0.08, label=f"deadband (±{dT}°C)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("T_zone − setpoint (°C)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Distribution of deviation from 21°C setpoint")

    ax = axes[2]
    for i in fan_idx:
        ax.plot(actions[:, i], lw=0.3, alpha=0.7, label=f"fan_{i}")
    ax2 = ax.twinx()
    for i in sat_idx:
        ax2.plot(actions[:, i], lw=0.3, alpha=0.5, ls="--", color="tab:red")
    ax.set_ylabel("Fan mass flow (kg/s)")
    ax2.set_ylabel("SAT setpoint (°C)", color="tab:red")
    ax.set_xlabel("Step")
    ax.set_title("Actuator time series (fan=solid, SAT=dashed red)")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[3]
    cum_temp = np.cumsum(temp_term)
    cum_energy = np.cumsum(energy_term)
    ax.plot(cum_temp, label="cum. temperature term", color="tab:blue")
    ax.plot(cum_energy, label="cum. energy term (w·E)", color="tab:orange")
    ax.plot(cum_temp + cum_energy, label="cum. total penalty", color="k", lw=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative penalty (reward units)")
    ax.set_title(
        f"Reward decomposition — total={reward_stats['total_reward']:.0f}, "
        f"temp share={reward_stats['share_of_penalty_from_temp']:.1%}, "
        f"energy share={reward_stats['share_of_penalty_from_energy']:.1%}"
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = output_dir / f"{stem}_analysis.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure → %s", fig_path)

    return {
        "building_id": meta["building_id"],
        "climate_zone": meta["climate_zone"],
        "temperatures": temp_stats,
        "actions": action_stats,
        "reward": reward_stats,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--buildings",
        nargs="+",
        required=True,
        help="List of <building_id>:<climate_zone> pairs, e.g. OfficeSmall-5167:1",
    )
    parser.add_argument(
        "--skip-rollouts",
        action="store_true",
        help="Re-analyze existing .npz files without re-running EnergyPlus.",
    )
    args = parser.parse_args()

    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, int]] = []
    for entry in args.buildings:
        bid, cz_str = entry.split(":")
        targets.append((bid, int(cz_str)))

    for bid, cz in targets:
        npz_path = out / f"{bid}_cz{cz}.npz"
        if not args.skip_rollouts or not npz_path.exists():
            run_rollout(bid, cz, out)

    full_report: list[dict[str, Any]] = []
    for bid, cz in targets:
        npz_path = out / f"{bid}_cz{cz}.npz"
        if not npz_path.exists():
            logger.warning("Missing %s — skipping", npz_path)
            continue
        full_report.append(analyze_one(npz_path, out))

    (out / "report.json").write_text(json.dumps(full_report, indent=2))
    logger.info("Report written → %s", out / "report.json")


if __name__ == "__main__":
    main()
