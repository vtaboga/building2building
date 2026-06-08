"""Shared helpers for reward-normalizer calibration rollouts and aggregation."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import building2building as b2b

logger = logging.getLogger(__name__)

CALIBRATION_TASK = "task_occ_emed"
CALIBRATION_SPLIT = "train"
SCHEMA_VERSION = 1
DEFAULT_RUN_PERIODS: tuple[str, ...] = ("winter", "summer", "full_year")

DEFAULT_BUILDING_TYPES: tuple[str, ...] = (
    "OfficeSmall",
    "OfficeMedium",
    "RestaurantFastFood",
    "RetailStandalone",
    "Warehouse",
    "SingleFamilyHouse",
)

RolloutFn = Callable[["RolloutSpec"], str]


@dataclass(frozen=True)
class RolloutSpec:
    """One ``(run_period, building_type, building_id)`` rollout."""

    run_period: str
    building_type: str
    building_id: str
    out_path: Path

    @property
    def key(self) -> str:
        return f"{self.run_period}/{self.building_type}/{self.building_id}"


@dataclass(frozen=True)
class BuildingStats:
    """Per-building mean deadband penalties for one run period."""

    run_period: str
    building_type: str
    building_id: str
    cz_key: str
    mean_temp_penalty: float
    mean_power_penalty: float
    n_steps: int
    seed: int | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "run_period": self.run_period,
            "building_type": self.building_type,
            "building_id": self.building_id,
            "cz_key": self.cz_key,
            "mean_temp_penalty": self.mean_temp_penalty,
            "mean_power_penalty": self.mean_power_penalty,
            "n_steps": self.n_steps,
        }
        if self.seed is not None:
            out["seed"] = self.seed
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BuildingStats:
        seed_raw = data.get("seed")
        return cls(
            run_period=str(data["run_period"]),
            building_type=str(data["building_type"]),
            building_id=str(data["building_id"]),
            cz_key=str(data["cz_key"]),
            mean_temp_penalty=float(data["mean_temp_penalty"]),
            mean_power_penalty=float(data["mean_power_penalty"]),
            n_steps=int(data["n_steps"]),
            seed=int(seed_raw) if seed_raw is not None else None,
        )


def default_data_dir() -> Path:
    """Return ``$SCRATCH/b2b/reward_normalizers/data`` or a ``/tmp`` fallback."""
    scratch = os.environ.get("SCRATCH")
    root = Path(scratch) if scratch else Path("/tmp")
    return root / "b2b" / "reward_normalizers" / "data"


def cz_key_for(building_type: str, building_id: str) -> str:
    """``cz0`` for CZ-less types, else ``cz<N>``."""
    if building_type in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
        return "cz0"
    return f"cz{b2b.get_climate_zone(building_type, building_id)}"


def list_train_buildings(
    building_types: Iterable[str],
    max_per_type: int | None,
    *,
    climate_zone: int | None = None,
) -> list[tuple[str, str]]:
    """Deterministic train-split ``(building_type, building_id)`` pairs."""
    picks: list[tuple[str, str]] = []
    for bt in building_types:
        if climate_zone is not None and bt not in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
            ids = sorted(
                b2b.list_buildings_by_climate_zone(
                    bt, climate_zone, split=CALIBRATION_SPLIT
                )
            )
        else:
            ids = sorted(b2b.list_buildings(bt, split=CALIBRATION_SPLIT))
        if max_per_type is not None:
            ids = ids[:max_per_type]
        picks.extend((bt, bid) for bid in ids)
    return picks


def shard_picks(
    picks: list[tuple[str, str]], shard_index: int, shard_count: int
) -> list[tuple[str, str]]:
    """1-based sharding over buildings."""
    if shard_count <= 1:
        return picks
    if not 1 <= shard_index <= shard_count:
        raise ValueError(
            f"shard-index={shard_index} must be in [1, {shard_count}]"
        )
    n = len(picks)
    chunk = -(-n // shard_count)
    start = (shard_index - 1) * chunk
    end = min(start + chunk, n)
    return picks[start:end]


def build_specs(
    picks: list[tuple[str, str]], data_dir: Path, run_periods: Iterable[str]
) -> list[RolloutSpec]:
    specs: list[RolloutSpec] = []
    for run_period in run_periods:
        for bt, bid in picks:
            specs.append(
                RolloutSpec(
                    run_period=run_period,
                    building_type=bt,
                    building_id=bid,
                    out_path=data_dir / run_period / bt / f"{bid}.json",
                )
            )
    return specs


def run_rollouts(
    specs: list[RolloutSpec], n_workers: int, rollout_fn: RolloutFn
) -> dict[str, str]:
    """Run ``rollout_fn`` for each spec whose cache file is missing."""
    statuses: dict[str, str] = {}
    pending = [s for s in specs if not s.out_path.exists()]
    skipped = len(specs) - len(pending)
    if skipped:
        logger.info("Skipping %d already-cached rollouts.", skipped)
    if not pending:
        return statuses

    if n_workers <= 1:
        for i, spec in enumerate(pending, 1):
            logger.info("[%d/%d] %s", i, len(pending), spec.key)
            statuses[spec.key] = rollout_fn(spec)
            logger.info("    -> %s", statuses[spec.key])
        return statuses

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(rollout_fn, s): s for s in pending}
        done = 0
        for fut in as_completed(futures):
            spec = futures[fut]
            done += 1
            try:
                statuses[spec.key] = fut.result()
            except Exception as exc:
                statuses[spec.key] = f"fail: worker: {exc}"
            logger.info(
                "[%d/%d] %s -> %s",
                done,
                len(pending),
                spec.key,
                statuses[spec.key],
            )
    return statuses


def load_stats(data_dir: Path, run_periods: Iterable[str]) -> list[BuildingStats]:
    out: list[BuildingStats] = []
    for run_period in run_periods:
        period_dir = data_dir / run_period
        for path in sorted(period_dir.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
                stats = BuildingStats.from_json(data)
                if stats.run_period != run_period:
                    logger.warning(
                        "Skipping %s: run_period=%s != directory %s",
                        path,
                        stats.run_period,
                        run_period,
                    )
                    continue
                out.append(stats)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", path, exc)
    return out


def aggregate(
    stats: list[BuildingStats],
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Median + IQR per ``(run_period, building_type, cz_key)`` bucket."""
    grouped: dict[str, dict[tuple[str, str], list[BuildingStats]]] = {}
    for s in stats:
        period_group = grouped.setdefault(s.run_period, {})
        period_group.setdefault((s.building_type, s.cz_key), []).append(s)

    aggregated: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for run_period, period_group in grouped.items():
        aggregated[run_period] = {}
        for key, members in period_group.items():
            tau_T_vals = [m.mean_temp_penalty for m in members]
            tau_E_vals = [m.mean_power_penalty for m in members]
            aggregated[run_period][key] = {
                "tau_T": float(statistics.median(tau_T_vals)),
                "tau_E": float(statistics.median(tau_E_vals)),
                "tau_T_iqr": _iqr(tau_T_vals),
                "tau_E_iqr": _iqr(tau_E_vals),
                "n_buildings": len(members),
                "_tau_T_vals": tau_T_vals,
                "_tau_E_vals": tau_E_vals,
            }
    return aggregated


def strip_internal_keys(
    aggregated: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    return {
        run_period: {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in period_payload.items()
        }
        for run_period, period_payload in aggregated.items()
    }


def write_reward_normalizers_yaml(
    aggregated: dict[str, dict[tuple[str, str], dict[str, Any]]],
    out_path: Path,
    run_periods: Iterable[str],
    *,
    generator_module: str,
    source_lines: list[str],
) -> None:
    header = (
        f"# Auto-generated by {generator_module}\n"
        f"# Do not edit by hand: regenerate via\n"
        f"#   python -m {generator_module} --mode aggregate\n"
        "\n"
        f"schema_version: {SCHEMA_VERSION}\n"
        "source:\n"
    )
    header += "".join(f"  {line}\n" for line in source_lines)
    sha = _git_sha()
    if sha is not None:
        header += f"  git_sha: {sha}\n"
    version = _b2b_version()
    if version is not None:
        header += f"  b2b_version: '{version}'\n"
    header += (
        "floor:\n"
        "  epsilon_abs: 1.0e-6\n"
        "  epsilon_rel: 0.05\n"
        "constants:\n"
    )

    body_lines: list[str] = []
    for run_period in run_periods:
        period_payload = aggregated.get(run_period, {})
        body_lines.append(f"  {run_period}:")
        if period_payload:
            body_lines.append(_format_period_constants_yaml(period_payload))
        else:
            body_lines.append("    {}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(body_lines) + "\n")


def save_calibration_plot(
    aggregated: dict[str, dict[tuple[str, str], dict[str, Any]]],
    plot_path: Path,
    run_periods: Iterable[str],
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    periods = list(run_periods)
    if not any(aggregated.get(period) for period in periods):
        logger.warning("No buckets aggregated; skipping calibration plot.")
        return

    fig, axes = plt.subplots(
        2,
        len(periods),
        figsize=(max(8, 0.35 * 40 * len(periods)), 8),
        squeeze=False,
    )
    colors = ("#B24A3F", "#1F4E79")
    row_titles = ("mean(temp_penalty) / tau_T", "mean(power_penalty) / tau_E")

    for col, run_period in enumerate(periods):
        period_payload = aggregated.get(run_period, {})
        bucket_keys = sorted(period_payload.keys())
        labels = [f"{bt}/{cz}" for bt, cz in bucket_keys]
        ratio_sets = [
            [
                [v / period_payload[k]["tau_T"] for v in period_payload[k]["_tau_T_vals"]]
                for k in bucket_keys
            ],
            [
                [v / period_payload[k]["tau_E"] for v in period_payload[k]["_tau_E_vals"]]
                for k in bucket_keys
            ],
        ]

        for row, ratios in enumerate(ratio_sets):
            ax = axes[row][col]
            if not bucket_keys:
                ax.set_title(f"{run_period}: no data")
                ax.axis("off")
                continue
            positions = list(range(1, len(labels) + 1))
            non_singleton = [
                (p, r) for p, r in zip(positions, ratios) if len(r) >= 2
            ]
            if non_singleton:
                pos_ne, r_ne = zip(*non_singleton)
                parts = ax.violinplot(
                    list(r_ne),
                    positions=list(pos_ne),
                    widths=0.7,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(colors[row])
                    body.set_edgecolor(colors[row])
                    body.set_alpha(0.4)
            for p, r in zip(positions, ratios):
                ax.scatter(
                    [p] * len(r),
                    r,
                    color=colors[row],
                    alpha=0.7,
                    s=12,
                    zorder=3,
                )
            ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.6)
            ax.set_title(f"{run_period}: {row_titles[row]}", fontsize=10)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def mean_deadband_penalties_from_infos(
    infos: list[dict[str, Any]],
    reward_fn: Any,
) -> tuple[float, float, int]:
    """Mean ``(temp_penalty, power_penalty)`` from post-step infos."""
    from building2building.simulator.rewards import _deadband_components

    temp_pen: list[float] = []
    power_pen: list[float] = []
    for info in infos:
        raw = info.get("raw_observation")
        if raw is None:
            continue
        tp, pp = _deadband_components(
            raw, reward_fn.controlled_zones, reward_fn.task_config, reward_fn.dT
        )
        temp_pen.append(tp)
        power_pen.append(pp)

    if not temp_pen:
        raise ValueError("no valid steps with raw_observation")

    return float(np.mean(temp_pen)), float(np.mean(power_pen)), len(temp_pen)


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return q3 - q1


def _format_period_constants_yaml(
    aggregated: dict[tuple[str, str], dict[str, Any]],
) -> str:
    by_type: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for (bt, cz_key), payload in aggregated.items():
        by_type.setdefault(bt, []).append((cz_key, payload))

    lines: list[str] = []
    for bt in sorted(by_type.keys()):
        lines.append(f"    {bt}:")
        for cz_key, payload in sorted(by_type[bt], key=lambda kv: kv[0]):
            lines.append(
                "      {cz}: {{ tau_T: {tau_T:.6g}, tau_E: {tau_E:.6g}, "
                "tau_T_iqr: {iqr_T:.6g}, tau_E_iqr: {iqr_E:.6g}, "
                "n_buildings: {n} }}".format(
                    cz=cz_key,
                    tau_T=payload["tau_T"],
                    tau_E=payload["tau_E"],
                    iqr_T=payload["tau_T_iqr"],
                    iqr_E=payload["tau_E_iqr"],
                    n=payload["n_buildings"],
                )
            )
    return "\n".join(lines)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def _b2b_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("building2building")
    except Exception:
        return None
