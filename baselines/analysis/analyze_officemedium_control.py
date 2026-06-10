#!/usr/bin/env python3
"""Diagnostic rollouts for the tuned OfficeMedium VAV air-loop controller.

This is the OfficeMedium counterpart to
:mod:`baselines.analysis.analyze_officesmall_control`.  Unlike the small
office, OfficeMedium uses a VAV air-loop system whose per-zone action
surface has three channels (flow fraction / damper, heating setpoint /
reheat, cooling setpoint) plus a single loop-level supply-air-temperature
setpoint; the tuned configurations live at
``baselines/configs/tuned_controllers/air_loop_officemedium_cz{cz}.yaml``.

The script runs a full-year deterministic rollout per building, records
per-zone temperatures, outdoor temperature, electricity + gas energy,
actions, reward, and a re-computed ``task1`` reward breakdown
(temperature term vs. energy term).  Trajectories are saved as compressed
``.npz`` files so the analysis step is fully offline and idempotent
(``--skip-rollouts``).

Usage::

    python -m baselines.analysis.analyze_officemedium_control \\
        --output-dir analysis/officemedium_tuned \\
        --buildings OfficeMedium-<id1>:1 OfficeMedium-<id4>:4 OfficeMedium-<id8>:8

Each argument follows the ``<building_id>:<climate_zone>`` convention
used by the small-office script; the climate zone is passed explicitly
because ``_get_climate_zone`` in ``run_reactive_control`` does not cope
with the hashed weather filenames currently in the registry.
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
from baselines.controllers.air_loop import AirLoopConfig, AirLoopPolicy

logger = logging.getLogger(__name__)

TUNED_CONFIGS_DIR = (
    Path(__file__).resolve().parent.parent / "configs" / "tuned_controllers"
)

TARGET_C = 21.0
DEADBAND_C = 1.0
ENERGY_WEIGHT = 0.01

# Threshold (in number of zones) above which the temperature panel shows
# a p5/p50/p95 envelope across zones instead of one line per zone.  For
# OfficeMedium models we have typically seen 10-15 conditioned zones, so
# 8 is a natural break point — still shows per-zone detail for small
# buildings but stays legible on the larger ones.
ZONE_ENVELOPE_THRESHOLD = 8


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass
class ActionBuckets:
    """Classification of flat action indices into VAV functional groups.

    The bucket layout is populated from the bound :class:`AirLoopPolicy`
    so that the analysis does not rely on regex matching of actuator
    names.  Lists are parallel to ``zone_names`` whenever they are
    per-zone.  ``cooling`` entries may be ``None`` for zones that do not
    expose an explicit cooling setpoint actuator.
    """

    sat: list[int]
    flow: list[int]
    reheat: list[int]
    cooling: list[int | None]
    zone_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sat": list(self.sat),
            "flow": list(self.flow),
            "reheat": list(self.reheat),
            "cooling": [None if c is None else int(c) for c in self.cooling],
            "zone_names": list(self.zone_names),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ActionBuckets:
        return cls(
            sat=[int(x) for x in raw["sat"]],
            flow=[int(x) for x in raw["flow"]],
            reheat=[int(x) for x in raw["reheat"]],
            cooling=[None if c is None else int(c) for c in raw["cooling"]],
            zone_names=[str(z) for z in raw["zone_names"]],
        )


def _buckets_from_policy(policy: AirLoopPolicy) -> ActionBuckets:
    """Read the action-index buckets from a bound :class:`AirLoopPolicy`."""
    sat_idx: list[int] = []
    flow_idx: list[int] = []
    reheat_idx: list[int] = []
    cooling_idx: list[int | None] = []
    zone_names: list[str] = []
    for loop in policy._loops:  # pylint: disable=protected-access
        sat_idx.append(int(loop.sat_act_idx))
        for z in loop.zones:
            flow_idx.append(int(z.flow_act_idx))
            reheat_idx.append(int(z.htg_act_idx))
            cooling_idx.append(None if z.clg_act_idx is None else int(z.clg_act_idx))
            zone_names.append(str(z.zone_name))
    return ActionBuckets(
        sat=sat_idx,
        flow=flow_idx,
        reheat=reheat_idx,
        cooling=cooling_idx,
        zone_names=zone_names,
    )


# ---------------------------------------------------------------------------
# Config + reward
# ---------------------------------------------------------------------------


def _load_tuned_config(cz: int) -> AirLoopConfig:
    p = TUNED_CONFIGS_DIR / f"air_loop_officemedium_cz{cz}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Tuned config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    raw.pop("type", None)
    # ``target_schedule`` is not part of AirLoopConfig; it may be present
    # in older YAMLs from the reactive-control pipeline.
    raw.pop("target_schedule", None)
    sat_aware_flow = bool(raw.pop("sat_aware_flow", True))
    float_kwargs = {k: float(v) for k, v in raw.items()}
    return AirLoopConfig(sat_aware_flow=sat_aware_flow, **float_kwargs)


def _deadband_reward_breakdown(
    zone_temps: dict[str, float],
    target: float,
    dT: float,
    energy_electricity: float,
    energy_gas: float,
    energy_weight: float,
) -> tuple[float, float, float]:
    """Recompute ``task1`` reward per-step and split into (temp, energy, total).

    Mirrors :func:`building2building.simulator.rewards.deadband_reward_function`
    so we can attribute each step of penalty to temperature vs. energy.
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


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_rollout(
    building_id: str,
    climate_zone: int,
    output_dir: Path,
) -> dict[str, Any]:
    cfg = _load_tuned_config(climate_zone)
    logger.info(
        "Building %s (cz=%d) — running full-year rollout with tuned VAV config",
        building_id,
        climate_zone,
    )

    eplus_dir = output_dir / "eplus" / building_id
    eplus_dir.mkdir(parents=True, exist_ok=True)
    env = b2b.make_env(
        "OfficeMedium",
        building_id=building_id,
        task="task1",
        run_period="full_year",
        eplus_output_dir=eplus_dir,
    )
    controlled_zones: list[str] = list(env.metadata["controlled_zones"])
    action_names: list[str] = list(env.metadata["action_names"])

    try:
        policy = AirLoopPolicy(cfg)
        policy.bind_env(env)
        buckets = _buckets_from_policy(policy)

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
        "action_buckets": buckets.to_dict(),
        "target_c": TARGET_C,
        "deadband_c": DEADBAND_C,
        "energy_weight": ENERGY_WEIGHT,
        "episode_steps": int(actions_arr.shape[0]),
        "total_reward": float(sum(reward_hist)),
        "sum_temp_term": float(sum(temp_term_hist)),
        "sum_energy_term": float(sum(energy_term_hist)),
        "tuned_config": {
            "reheat_sp_min": cfg.reheat_sp_min,
            "reheat_sp_max": cfg.reheat_sp_max,
            "flow_min": cfg.flow_min,
            "flow_max": cfg.flow_max,
            "sat_min": cfg.sat_min,
            "sat_max": cfg.sat_max,
            "target_temp": cfg.target_temp,
        },
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


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _temperature_stats(
    dev: np.ndarray,
    zones: list[str],
    dT: float,
) -> dict[str, Any]:
    per_zone: list[dict[str, Any]] = []
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
    return {
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


def _actuator_stat(
    name: str, values: np.ndarray, extra: dict[str, float] | None = None
) -> dict[str, Any]:
    base = {
        "name": name,
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }
    if extra is not None:
        base.update({k: float(v) for k, v in extra.items()})
    return base


def _action_stats(
    actions: np.ndarray,
    action_names: list[str],
    buckets: ActionBuckets,
    tuned_cfg: dict[str, float],
) -> dict[str, Any]:
    """Compute per-bucket actuator statistics + a fleet-level VAV summary."""
    flow_min_cfg = float(tuned_cfg.get("flow_min", 0.0))
    flow_max_cfg = float(tuned_cfg.get("flow_max", 1.0))
    reheat_sp_min_cfg = float(tuned_cfg.get("reheat_sp_min", 10.0))
    flow_eps = 1e-3
    reheat_eps = 1e-2

    flow_stats: list[dict[str, Any]] = []
    fracs_at_min: list[float] = []
    fracs_at_max: list[float] = []
    for act_idx, zone in zip(buckets.flow, buckets.zone_names):
        v = actions[:, act_idx]
        f_at_min = float(np.mean(v <= flow_min_cfg + flow_eps))
        f_at_max = float(np.mean(v >= flow_max_cfg - flow_eps))
        flow_stats.append(
            _actuator_stat(
                f"{zone}::{action_names[act_idx]}",
                v,
                {"frac_at_min": f_at_min, "frac_at_max": f_at_max},
            )
        )
        fracs_at_min.append(f_at_min)
        fracs_at_max.append(f_at_max)

    reheat_stats: list[dict[str, Any]] = []
    reheat_active_fracs: list[float] = []
    for act_idx, zone in zip(buckets.reheat, buckets.zone_names):
        v = actions[:, act_idx]
        frac_above_min = float(np.mean(v > reheat_sp_min_cfg + reheat_eps))
        reheat_stats.append(
            _actuator_stat(
                f"{zone}::{action_names[act_idx]}",
                v,
                {"frac_above_reheat_sp_min": frac_above_min},
            )
        )
        reheat_active_fracs.append(frac_above_min)

    sat_stats = [_actuator_stat(action_names[i], actions[:, i]) for i in buckets.sat]

    saturation_threshold = 0.5
    n_flow = len(buckets.flow)
    frac_zones_saturated = (
        float(
            np.mean(
                [
                    (fmin + fmax) >= saturation_threshold
                    for fmin, fmax in zip(fracs_at_min, fracs_at_max)
                ]
            )
        )
        if n_flow > 0
        else float("nan")
    )

    vav_summary = {
        "n_zones": n_flow,
        "mean_frac_at_min": (
            float(np.mean(fracs_at_min)) if n_flow > 0 else float("nan")
        ),
        "mean_frac_at_max": (
            float(np.mean(fracs_at_max)) if n_flow > 0 else float("nan")
        ),
        "frac_zones_saturated": frac_zones_saturated,
        "mean_frac_reheat_active": (
            float(np.mean(reheat_active_fracs)) if reheat_active_fracs else float("nan")
        ),
    }

    return {
        "n_steps": int(actions.shape[0]),
        "flow_actuators": flow_stats,
        "reheat_actuators": reheat_stats,
        "sat_actuators": sat_stats,
        "vav_summary": vav_summary,
    }


def _reward_stats(
    rewards: np.ndarray,
    temp_term: np.ndarray,
    energy_term: np.ndarray,
    outdoor: np.ndarray,
) -> dict[str, Any]:
    total_penalty = float(temp_term.sum() + energy_term.sum())
    if total_penalty > 0:
        temp_share = float(temp_term.sum() / total_penalty)
        energy_share = float(energy_term.sum() / total_penalty)
    else:
        temp_share = float("nan")
        energy_share = float("nan")
    return {
        "total_reward": float(rewards.sum()),
        "mean_reward_per_step": float(rewards.mean()),
        "sum_temp_term": float(temp_term.sum()),
        "sum_energy_term": float(energy_term.sum()),
        "mean_temp_term_per_step": float(temp_term.mean()),
        "mean_energy_term_per_step": float(energy_term.mean()),
        "share_of_penalty_from_temp": temp_share,
        "share_of_penalty_from_energy": energy_share,
        "outdoor_temperature_c": {
            "mean": float(outdoor.mean()),
            "min": float(outdoor.min()),
            "max": float(outdoor.max()),
        },
    }


def _plot(
    npz_path: Path,
    output_dir: Path,
    meta: dict[str, Any],
    temps: np.ndarray,
    outdoor: np.ndarray,
    actions: np.ndarray,
    temp_term: np.ndarray,
    energy_term: np.ndarray,
    buckets: ActionBuckets,
    reward_stats: dict[str, Any],
) -> Path:
    zones: list[str] = meta["controlled_zones"]
    target = float(meta["target_c"])
    dT = float(meta["deadband_c"])
    dev = temps - target

    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=False)
    stem = npz_path.stem
    hours = np.arange(temps.shape[0]) * (60.0 / 12.0) / 60.0  # 12 ts/hour

    # (1) Zone temperatures ------------------------------------------------
    ax = axes[0]
    n_zones = len(zones)
    if n_zones <= ZONE_ENVELOPE_THRESHOLD:
        for i, z in enumerate(zones):
            ax.plot(hours, temps[:, i], lw=0.4, alpha=0.7, label=z)
    else:
        p05 = np.percentile(temps, 5, axis=1)
        p50 = np.percentile(temps, 50, axis=1)
        p95 = np.percentile(temps, 95, axis=1)
        ax.fill_between(hours, p05, p95, alpha=0.2, label="p5-p95 across zones")
        ax.plot(hours, p50, lw=0.5, label="median zone")
    ax.axhline(target, color="k", lw=1, ls="--", label=f"setpoint={target}°C")
    ax.axhline(target + dT, color="grey", lw=0.6, ls=":", label=f"±{dT}°C deadband")
    ax.axhline(target - dT, color="grey", lw=0.6, ls=":")
    ax.plot(hours, outdoor, lw=0.3, color="tab:brown", alpha=0.5, label="outdoor")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Hour of year")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title(
        f"{meta['building_id']} (cz={meta['climate_zone']}) — zone temperatures"
        f" ({n_zones} zones)"
    )

    # (2) Deviation histogram ---------------------------------------------
    ax = axes[1]
    bins = np.linspace(-12, 12, 97)
    if n_zones <= ZONE_ENVELOPE_THRESHOLD:
        for i, z in enumerate(zones):
            ax.hist(
                dev[:, i],
                bins=bins,
                alpha=0.35,
                label=z,
                density=True,
                histtype="stepfilled",
            )
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.hist(
            dev.reshape(-1),
            bins=bins,
            alpha=0.5,
            label="all zones (pooled)",
            density=True,
            histtype="stepfilled",
        )
        ax.legend(fontsize=7)
    ax.axvspan(-dT, dT, color="green", alpha=0.08, label=f"deadband (±{dT}°C)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("T_zone − setpoint (°C)")
    ax.set_ylabel("density")
    ax.set_title("Distribution of deviation from 21°C setpoint")

    # (3) Flow actuators ---------------------------------------------------
    ax = axes[2]
    for act_idx, zone in zip(buckets.flow, buckets.zone_names):
        ax.plot(actions[:, act_idx], lw=0.3, alpha=0.6, label=zone)
    ax.set_ylabel("Flow fraction / mass flow (raw actuator units)")
    ax.set_xlabel("Step")
    ax.set_title(f"Per-zone VAV flow actuators ({len(buckets.flow)} zones)")
    if len(buckets.flow) <= ZONE_ENVELOPE_THRESHOLD:
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    # (4) Reheat SP + loop SAT overlay ------------------------------------
    ax = axes[3]
    for act_idx, zone in zip(buckets.reheat, buckets.zone_names):
        ax.plot(actions[:, act_idx], lw=0.3, alpha=0.5, label=f"reheat {zone}")
    ax.set_ylabel("Reheat SP (°C)")
    ax.set_xlabel("Step")
    ax2 = ax.twinx()
    for act_idx in buckets.sat:
        ax2.plot(
            actions[:, act_idx],
            lw=0.5,
            alpha=0.9,
            color="tab:red",
            ls="--",
            label=f"SAT {action_name_short(meta['action_names'][act_idx])}",
        )
    ax2.set_ylabel("SAT setpoint (°C)", color="tab:red")
    ax.set_title(f"Reheat setpoints (left) + loop SAT setpoint (right, dashed red)")
    if len(buckets.reheat) <= ZONE_ENVELOPE_THRESHOLD:
        ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")

    # (5) Cumulative reward decomposition ---------------------------------
    ax = axes[4]
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
    return fig_path


def action_name_short(name: str) -> str:
    """Keep only the last ``::`` component for legend readability."""
    parts = name.split("::")
    return parts[-1] if parts else name


def analyze_one(
    npz_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute quantitative stats + plots for a single building rollout."""
    data = np.load(npz_path, allow_pickle=False)
    meta: dict[str, Any] = json.loads(str(data["meta"]))
    zones: list[str] = meta["controlled_zones"]
    action_names: list[str] = meta["action_names"]
    buckets = ActionBuckets.from_dict(meta["action_buckets"])
    tuned_cfg: dict[str, float] = meta.get("tuned_config", {})
    target = float(meta["target_c"])
    dT = float(meta["deadband_c"])

    temps: np.ndarray = data["zone_temperatures"]
    outdoor: np.ndarray = data["outdoor_temperature"]
    actions: np.ndarray = data["actions"]
    rewards: np.ndarray = data["rewards"]
    temp_term: np.ndarray = data["temp_term"]
    energy_term: np.ndarray = data["energy_term"]

    dev = temps - target
    temp_stats = _temperature_stats(dev, zones, dT)
    action_stats = _action_stats(actions, action_names, buckets, tuned_cfg)
    reward_stats = _reward_stats(rewards, temp_term, energy_term, outdoor)

    fig_path = _plot(
        npz_path,
        output_dir,
        meta,
        temps,
        outdoor,
        actions,
        temp_term,
        energy_term,
        buckets,
        reward_stats,
    )
    logger.info("Saved figure → %s", fig_path)

    return {
        "building_id": meta["building_id"],
        "climate_zone": meta["climate_zone"],
        "temperatures": temp_stats,
        "actions": action_stats,
        "reward": reward_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        help=(
            "List of <building_id>:<climate_zone> pairs, e.g. " "OfficeMedium-5167:1"
        ),
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
