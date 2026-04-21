"""Aggregate per-building reports from the v2 tuned-controller analysis.

Consumes the artefacts emitted by ``baselines/analyze_tuned_controller.py``
(``*.report.json`` for fast aggregation, ``*.npz`` only when diagnosing worst
performers) and writes:

* ``analysis/<type>_tuned_v2/SUMMARY.md`` — one per building type, with
  per-CZ and overall mean / p50 / p95 for reward, comfort violation, and
  energy, plus the top-3 best / bottom-3 worst buildings by reward.
* ``analysis/SUMMARY_WORST_PERFORMERS.md`` — the ~5 globally worst
  buildings across all types, with a control-vs-HVAC diagnosis using
  the trajectory arrays in the matching ``.npz`` file.

Run from the repo root::

    python -m baselines.analysis_tools.summarize_tuned_v2

The script is idempotent: it only *reads* the per-building artefacts and
(re-)writes the SUMMARY markdowns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"

TYPE_DIRS: dict[str, str] = {
    "SingleFamilyHouse": "singlefamilyhouse_tuned_v2",
    "RestaurantFastFood": "restaurantfastfood_tuned_v2",
    "OfficeMedium": "officemedium_tuned_v2",
    "OfficeSmall": "officesmall_tuned_v2",
}

# Worst-performer heuristics (absolute thresholds, see diagnose() below).
SATURATION_FRAC_THRESHOLD: float = 0.50
COMFORT_VIOLATION_FRAC_THRESHOLD: float = 0.05
WARMUP_FRAC: float = 0.10


@dataclass(frozen=True)
class BuildingRecord:
    """One row per completed rollout."""

    building_type: str
    building_id: str
    climate_zone: int
    episode_length: int
    reward_mean_per_step: float
    reward_total: float
    energy_term_mean: float  # Wh/m² per step (elec + gas)
    temp_term_mean: float  # MSE(T − target) averaged over controlled zones & time
    frac_in_deadband: float
    mean_dev_c: float
    pct_95_abs_dev: float
    max_dev_c: float
    report_path: Path
    npz_path: Path

    @property
    def comfort_violation_frac(self) -> float:
        return max(0.0, 1.0 - self.frac_in_deadband)

    @property
    def energy_wh_per_m2_year(self) -> float:
        return self.energy_term_mean * self.episode_length


def _load_reports(type_dir: Path) -> list[BuildingRecord]:
    records: list[BuildingRecord] = []
    for rp in sorted(type_dir.glob("*.report.json")):
        try:
            r = json.loads(rp.read_text())
        except json.JSONDecodeError:
            logger.exception("Bad JSON: %s", rp)
            continue
        agg = r.get("temperatures", {}).get("aggregate", {}) or {}
        if not agg:
            logger.warning(
                "Skipping %s — no aggregate temperatures (empty per-zone report)",
                rp.name,
            )
            continue
        p95 = max(abs(agg.get("pct_95", 0.0)), abs(agg.get("pct_5", 0.0)))
        max_dev = max(
            abs(agg.get("max_over_c", 0.0)), abs(agg.get("max_under_c", 0.0))
        )
        records.append(
            BuildingRecord(
                building_type=r["building_type"],
                building_id=r["building_id"],
                climate_zone=int(r["climate_zone"]),
                episode_length=int(r["episode_length"]),
                reward_mean_per_step=float(r["reward"]["mean_per_step"]),
                reward_total=float(r["reward"]["total"]),
                energy_term_mean=float(r["reward"]["energy_term_mean"]),
                temp_term_mean=float(r["reward"]["temp_term_mean"]),
                frac_in_deadband=float(agg["frac_in_deadband"]),
                mean_dev_c=float(agg["mean_dev_c"]),
                pct_95_abs_dev=float(p95),
                max_dev_c=float(max_dev),
                report_path=rp,
                npz_path=rp.with_suffix("").with_suffix(".npz"),
            )
        )
    return records


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _fmt_row(label: str, stats: dict[str, float]) -> str:
    return (
        f"| {label} | {stats['n']} | "
        f"{stats['mean']:+.3f} | {stats['p50']:+.3f} | {stats['p95']:+.3f} |"
    )


def _per_cz_table(
    records: list[BuildingRecord],
    attr: str,
    *,
    title: str,
    subtitle: str,
    fmt_spec: str = "+.4f",
) -> str:
    by_cz: dict[int, list[float]] = {}
    for r in records:
        by_cz.setdefault(r.climate_zone, []).append(float(getattr(r, attr)))
    lines = [f"### {title}", "", subtitle, ""]
    lines.append("| Climate zone | N | mean | p50 | p95 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for cz in sorted(by_cz):
        s = _summary_stats(by_cz[cz])
        lines.append(
            f"| cz{cz} | {s['n']} | {s['mean']:{fmt_spec}} "
            f"| {s['p50']:{fmt_spec}} | {s['p95']:{fmt_spec}} |"
        )
    overall = _summary_stats([float(getattr(r, attr)) for r in records])
    lines.append(
        f"| **overall** | **{overall['n']}** | **{overall['mean']:{fmt_spec}}** "
        f"| **{overall['p50']:{fmt_spec}}** | **{overall['p95']:{fmt_spec}}** |"
    )
    return "\n".join(lines)


def _leaderboard(
    records: list[BuildingRecord], *, k: int
) -> tuple[list[BuildingRecord], list[BuildingRecord]]:
    """Return (best-k, worst-k) by ``reward_mean_per_step`` (higher = better)."""
    by_reward = sorted(records, key=lambda r: r.reward_mean_per_step, reverse=True)
    return by_reward[:k], list(reversed(by_reward[-k:]))


def _leaderboard_table(
    records: list[BuildingRecord], *, title: str
) -> str:
    lines = [f"### {title}", ""]
    lines.append(
        "| Building | CZ | reward/step | 1 − frac_in_deadband | p95 |abs dev| (°C) | max |dev| (°C) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in records:
        lines.append(
            f"| `{r.building_id}` | {r.climate_zone} | "
            f"{r.reward_mean_per_step:+.4f} | {r.comfort_violation_frac:.3f} "
            f"| {r.pct_95_abs_dev:.2f} | {r.max_dev_c:.2f} |"
        )
    return "\n".join(lines)


def write_per_type_summary(bt: str, records: list[BuildingRecord]) -> Path:
    out_dir = ANALYSIS_ROOT / TYPE_DIRS[bt]
    out_path = out_dir / "SUMMARY.md"
    body: list[str] = [
        f"# {bt} — tuned reactive controller, full-year test rollouts",
        "",
        f"Aggregated from **{len(records)}** per-building reports under ",
        f"`analysis/{TYPE_DIRS[bt]}/`. Metrics:",
        "",
        "* **reward/step** — `reward.mean_per_step` from `*.report.json` "
        "(reward = −[MSE(T − target) + elec + gas], higher is better).",
        "* **comfort violation** — `1 − frac_in_deadband` on controlled zones "
        "(deadband half-width = 1°C).",
        "* **energy** — `energy_term_mean × episode_length` in Wh/m²·year "
        "(HVAC electricity + gas).",
        "",
    ]
    body.append(
        _per_cz_table(
            records,
            "reward_mean_per_step",
            title="Reward per step",
            subtitle="Higher is better.  Upper-bounded by 0.",
        )
    )
    body.append("")
    body.append(
        _per_cz_table(
            records,
            "comfort_violation_frac",
            title="Comfort violation fraction",
            subtitle="Fraction of timesteps with |T − setpoint| > 1°C on controlled zones.  Lower is better.",
            fmt_spec=".4f",
        )
    )
    body.append("")
    body.append(
        _per_cz_table(
            records,
            "energy_wh_per_m2_year",
            title="Annual HVAC energy (Wh/m²)",
            subtitle="HVAC electricity + natural gas, summed across the 105120-step episode.",
            fmt_spec=".1f",
        )
    )
    body.append("")
    best, worst = _leaderboard(records, k=3)
    body.append(_leaderboard_table(best, title="Top-3 best by reward/step"))
    body.append("")
    body.append(_leaderboard_table(worst, title="Bottom-3 worst by reward/step"))
    body.append("")
    out_path.write_text("\n".join(body))
    return out_path


# ---------------------------------------------------------------------------
# Worst-performer diagnosis (uses the .npz trajectory)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnosis:
    """Structured diagnosis for a single worst-performing building."""

    verdict: Literal["hvac_limited", "policy_suboptimal", "warmup_artefact", "mixed"]
    actuator_saturation_max: float
    comfort_violation_frac: float
    mean_abs_action_delta: list[float]
    warmup_reward_ratio: float  # first-WARMUP_FRAC vs rest mean reward
    setpoint_track_during_saturation: float  # mean |dev| when any fan is saturated
    notes: list[str]


def _diagnose_worst(record: BuildingRecord) -> Diagnosis:
    """Inspect the trajectory npz to classify failure mode."""
    if not record.npz_path.exists():
        return Diagnosis(
            verdict="policy_suboptimal",
            actuator_saturation_max=float("nan"),
            comfort_violation_frac=record.comfort_violation_frac,
            mean_abs_action_delta=[],
            warmup_reward_ratio=float("nan"),
            setpoint_track_during_saturation=float("nan"),
            notes=[f"Missing npz at {record.npz_path}"],
        )

    data = np.load(record.npz_path, allow_pickle=False)
    actions = data["actions"]  # (T-1, n_act)
    zone_t = data["zone_temperatures"]  # (T, n_zones)
    setpoints = data["setpoints"]  # (T, n_zones)
    rewards = data["rewards"]  # (T-1,)
    meta = json.loads(str(data["meta"]))
    action_names: list[str] = meta.get("action_names", [])
    zone_names: list[str] = meta.get("zone_names", [])
    controlled_zones = set(meta.get("controlled_zones", []))

    # Actuator saturation: fraction of steps at the per-actuator min/max.
    a_min = actions.min(axis=0)
    a_max = actions.max(axis=0)
    frac_min = (actions <= a_min + 1e-6).mean(axis=0)
    frac_max = (actions >= a_max - 1e-6).mean(axis=0)
    sat_by_actuator = np.maximum(frac_min, frac_max)
    saturation_max = float(sat_by_actuator.max())

    # Thrashing: mean abs Δaction per actuator, scaled by actuator range.
    deltas = np.abs(np.diff(actions, axis=0))
    rng = np.maximum(a_max - a_min, 1e-6)
    mean_abs_norm_delta = (deltas.mean(axis=0) / rng).tolist()

    # Warmup artefact check: compare first-WARMUP_FRAC of rewards to the rest.
    n = len(rewards)
    warmup_len = max(1, int(n * WARMUP_FRAC))
    warmup_mean = float(rewards[:warmup_len].mean())
    steady_mean = float(rewards[warmup_len:].mean())
    warmup_ratio = warmup_mean / (steady_mean if abs(steady_mean) > 1e-9 else 1.0)

    # For unitary-HVAC setups, two actuators control each AHU:
    #
    #   * ``Fan::Fan Air Mass Flow Rate`` — at max → fan delivering all the
    #     air it can;
    #   * ``Schedule:Constant::Schedule Value::... temp setpoint schedule``
    #     — at min → thermostat commanded for maximum cooling.
    #
    # Either saturation regime means "controller is asking for more cooling
    # than the system can provide".  We therefore count a step as
    # ``max_cooling_demand`` if *any* fan is at its max OR any setpoint
    # actuator is at its min.
    fan_indices = [
        i for i, n in enumerate(action_names) if "Fan Air Mass Flow Rate" in n
    ]
    sp_indices = [
        i for i, n in enumerate(action_names) if "setpoint schedule" in n.lower()
    ]
    controlled_cols = [
        j for j, z in enumerate(zone_names) if z in controlled_zones
    ]
    max_cooling_mask = np.zeros(actions.shape[0], dtype=bool)
    for fi in fan_indices:
        col = actions[:, fi]
        max_cooling_mask |= col >= col.max() - 1e-6
    for si in sp_indices:
        col = actions[:, si]
        max_cooling_mask |= col <= col.min() + 1e-6
    max_heating_mask = np.zeros(actions.shape[0], dtype=bool)
    for fi in fan_indices:
        col = actions[:, fi]
        max_heating_mask |= col >= col.max() - 1e-6
    for si in sp_indices:
        col = actions[:, si]
        max_heating_mask |= col >= col.max() - 1e-6
    if controlled_cols:
        dev = zone_t[1:, controlled_cols] - setpoints[1:, controlled_cols]
        abs_dev_when_sat = (
            float(np.abs(dev[max_cooling_mask | max_heating_mask]).mean())
            if (max_cooling_mask | max_heating_mask).any()
            else 0.0
        )
    else:
        abs_dev_when_sat = float("nan")

    # Sign of the deviation tells us the direction of the HVAC limit.
    if controlled_cols:
        dev_all = zone_t[1:, controlled_cols] - setpoints[1:, controlled_cols]
        frac_too_hot = float((dev_all > 1.0).mean())
        frac_too_cold = float((dev_all < -1.0).mean())
    else:
        frac_too_hot = frac_too_cold = 0.0

    # ── Classify verdict ────────────────────────────────────────────────
    notes: list[str] = []
    # HVAC-limited: fan actuator saturates for >threshold of the year AND
    # deviation remains >1°C while saturated.
    hvac_limited = (
        saturation_max >= SATURATION_FRAC_THRESHOLD
        and abs_dev_when_sat > 1.0
    )
    # Warmup artefact: first WARMUP_FRAC of year is >2× worse than steady state.
    warmup_dominates = (
        warmup_mean < steady_mean
        and abs(warmup_ratio) > 2.0
        and (steady_mean - warmup_mean) / max(abs(steady_mean), 1e-9) > 0.5
    )
    # Thrashing: any actuator has mean norm Δ > 0.5 (i.e. swings ≥ half its range
    # every step on average).
    thrashing = any(d > 0.5 for d in mean_abs_norm_delta)

    if hvac_limited and record.comfort_violation_frac > COMFORT_VIOLATION_FRAC_THRESHOLD:
        verdict: Literal["hvac_limited", "policy_suboptimal", "warmup_artefact", "mixed"] = "hvac_limited"
        direction = (
            "undersized cooling — zone overheats"
            if frac_too_hot > frac_too_cold
            else "undersized heating — zone underheats"
            if frac_too_cold > frac_too_hot
            else "mixed heating/cooling shortfall"
        )
        notes.append(
            f"Some actuator saturates at its extreme on {saturation_max:.0%} "
            f"of the year (fan-at-max OR setpoint-at-min → 'demand max cooling'); "
            f"mean |T−setpoint| during those steps is {abs_dev_when_sat:.2f}°C. "
            f"→ {direction} — HVAC capacity is the bottleneck, not the controller."
        )
        notes.append(
            f"Directional bias: {frac_too_hot:.0%} of steps are ≥1°C above setpoint, "
            f"{frac_too_cold:.0%} are ≥1°C below."
        )
    elif warmup_dominates:
        verdict = "warmup_artefact"
        notes.append(
            f"First {WARMUP_FRAC:.0%} of the year has mean reward "
            f"{warmup_mean:+.3f} vs {steady_mean:+.3f} in steady state "
            f"(ratio {warmup_ratio:.1f}) → simulation warmup dominates the score."
        )
    elif thrashing:
        verdict = "policy_suboptimal"
        offenders = [
            action_names[i].split("::")[0]
            for i, d in enumerate(mean_abs_norm_delta)
            if d > 0.5
        ]
        notes.append(
            f"Mean per-step norm Δaction exceeds 0.5 for actuators: {offenders} "
            "→ controller is thrashing (bang-bang oscillation)."
        )
    else:
        verdict = "policy_suboptimal"
        notes.append(
            f"No single clear bottleneck (max saturation={saturation_max:.0%}, "
            f"abs |dev| during saturation={abs_dev_when_sat:.2f}°C, "
            f"warmup-to-steady ratio={warmup_ratio:.2f}). "
            "Likely a policy-tuning limitation: the fixed setpoint schedule "
            "cannot exploit forecasted load without preview."
        )

    if record.pct_95_abs_dev > 2.0 and verdict == "policy_suboptimal":
        notes.append(
            f"p95 |T−setpoint| = {record.pct_95_abs_dev:.2f}°C — controller "
            "tolerates sustained ≥2°C excursions ~5% of the year."
        )

    return Diagnosis(
        verdict=verdict,
        actuator_saturation_max=saturation_max,
        comfort_violation_frac=record.comfort_violation_frac,
        mean_abs_action_delta=mean_abs_norm_delta,
        warmup_reward_ratio=warmup_ratio,
        setpoint_track_during_saturation=abs_dev_when_sat,
        notes=notes,
    )


def write_worst_performers(
    all_records: list[BuildingRecord], *, k: int = 5
) -> Path:
    worst = sorted(all_records, key=lambda r: r.reward_mean_per_step)[:k]
    out_path = ANALYSIS_ROOT / "SUMMARY_WORST_PERFORMERS.md"
    lines: list[str] = [
        "# Worst-performing buildings — tuned reactive controller, v2",
        "",
        f"Bottom {k} of {len(all_records)} buildings across all 4 types, "
        "ranked by `reward_mean_per_step` (lower = worse).  For each "
        "worst performer the trajectory in the matching `.npz` is inspected "
        "to answer: **is this a control-policy failure, or an HVAC/simulation "
        "limit?**",
        "",
        "Diagnostic thresholds:",
        "",
        f"* **HVAC-limited** — max per-actuator saturation fraction ≥ {SATURATION_FRAC_THRESHOLD:.0%} "
        "**and** mean |T−setpoint| during saturation > 1°C.",
        f"* **Warmup-artefact** — first {WARMUP_FRAC:.0%} of the year is "
        "≥2× worse than the rest.",
        "* **Policy-suboptimal / thrashing** — any actuator has mean per-step "
        "normalised |Δaction| > 0.5 (swings ≥ half its range every step).",
        "",
        "| # | Building | Type | CZ | reward/step | comfort viol | p95 |dev| | max |dev| | Verdict |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    diagnoses: list[tuple[BuildingRecord, Diagnosis]] = []
    for rank, r in enumerate(worst, 1):
        d = _diagnose_worst(r)
        diagnoses.append((r, d))
        lines.append(
            f"| {rank} | `{r.building_id}` | {r.building_type} | {r.climate_zone} "
            f"| {r.reward_mean_per_step:+.4f} | {r.comfort_violation_frac:.3f} "
            f"| {r.pct_95_abs_dev:.2f}°C | {r.max_dev_c:.2f}°C | **{d.verdict}** |"
        )
    lines.append("")
    lines.append("## Per-building diagnosis")
    lines.append("")
    for rank, (r, d) in enumerate(diagnoses, 1):
        rel_plot = (
            Path("analysis")
            / TYPE_DIRS[r.building_type]
            / f"{r.building_id}_cz{r.climate_zone}_analysis.png"
        )
        lines.append(f"### {rank}. `{r.building_id}` ({r.building_type}, cz{r.climate_zone}) — **{d.verdict}**")
        lines.append("")
        lines.append(f"* Reward/step: `{r.reward_mean_per_step:+.4f}`")
        lines.append(f"* Comfort violation (1 − frac_in_deadband): `{r.comfort_violation_frac:.3f}`")
        lines.append(f"* Max actuator saturation fraction: `{d.actuator_saturation_max:.3f}`")
        lines.append(
            f"* Mean |T−setpoint| during max-demand steps "
            f"(fan-at-max ∨ setpoint-at-extreme): "
            f"`{d.setpoint_track_during_saturation:.2f}°C`"
        )
        lines.append(
            f"* Reward ratio warmup/steady: "
            f"`{d.warmup_reward_ratio:.2f}` "
            f"(first {WARMUP_FRAC:.0%} vs rest of year)"
        )
        lines.append("")
        for note in d.notes:
            lines.append(f"  - {note}")
        lines.append("")
        lines.append(f"Plot: [`{rel_plot}`]({rel_plot})")
        lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    all_records: list[BuildingRecord] = []
    for bt, subdir in TYPE_DIRS.items():
        type_dir = ANALYSIS_ROOT / subdir
        records = _load_reports(type_dir)
        logger.info("%s: %d reports loaded", bt, len(records))
        if not records:
            continue
        out = write_per_type_summary(bt, records)
        logger.info("  wrote %s", out.relative_to(ANALYSIS_ROOT.parent))
        all_records.extend(records)

    worst_out = write_worst_performers(all_records, k=5)
    logger.info("wrote %s", worst_out.relative_to(ANALYSIS_ROOT.parent))


if __name__ == "__main__":
    main()
