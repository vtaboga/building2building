#!/usr/bin/env python3
"""Benchmark EnergyPlus simulation step times per building type.

Parses existing evaluation log files to extract ``EnergyPlus Run Time``
entries and computes per-step timing statistics.  For building types with
no available log data, runs a short benchmark (3 buildings, full year)
to gather the missing information.

The output is a LaTeX table saved to ``analysis/tables/step_time_table.tex``
and printed to stdout.

Usage::

    python scripts/benchmark_step_time.py
    python scripts/benchmark_step_time.py --run-missing
"""

from __future__ import annotations

import argparse
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
OUTPUT_TABLE_PATH = REPO_ROOT / "analysis" / "tables" / "step_time_table.tex"

STEPS_PER_EPISODE = 10001

BUILDING_TYPE_DISPLAY: dict[str, str] = {
    "OfficeSmall": "Office Small",
    "OfficeMedium": "Office Medium",
    "RetailStandalone": "Retail Standalone",
    "RestaurantFastFood": "Restaurant Fast Food",
    "Warehouse": "Warehouse",
    "SingleZone": "Single-Family House",
}

BUILDING_TYPE_ORDER = list(BUILDING_TYPE_DISPLAY.keys())

RUN_TIME_RE = re.compile(
    r"EnergyPlus Run Time=(\d+)hr\s+(\d+)min\s+([\d.]+)sec"
)

# Mapping from log filenames to building types, determined by inspecting
# the corresponding .err files for "Evaluating ... for <BuildingType>".
LOG_FILE_TO_BUILDING_TYPE: dict[str, str] = {
    "eval_g36_8864303_0.out": "OfficeSmall",
    "eval_g36_8864303_1.out": "RetailStandalone",
    "eval_g36_8864303_2.out": "RestaurantFastFood",
    "eval_g36_8864303_3.out": "Warehouse",
    "eval_officemedium_8878996.out": "OfficeMedium",
    "tune_g36_test_8876626.out": "SingleZone",
}


@dataclass
class TimingStats:
    building_type: str
    run_times_sec: list[float] = field(default_factory=list)
    n_steps: int = STEPS_PER_EPISODE
    source: str = ""

    @property
    def n_samples(self) -> int:
        return len(self.run_times_sec)

    @property
    def mean_total_sec(self) -> float:
        return float(np.mean(self.run_times_sec)) if self.run_times_sec else 0.0

    @property
    def std_total_sec(self) -> float:
        return float(np.std(self.run_times_sec)) if self.run_times_sec else 0.0

    @property
    def mean_step_ms(self) -> float:
        if not self.run_times_sec or self.n_steps == 0:
            return 0.0
        return self.mean_total_sec / self.n_steps * 1000.0

    @property
    def std_step_ms(self) -> float:
        if not self.run_times_sec or self.n_steps == 0:
            return 0.0
        return self.std_total_sec / self.n_steps * 1000.0

    @property
    def min_step_ms(self) -> float:
        if not self.run_times_sec or self.n_steps == 0:
            return 0.0
        return float(np.min(self.run_times_sec)) / self.n_steps * 1000.0

    @property
    def max_step_ms(self) -> float:
        if not self.run_times_sec or self.n_steps == 0:
            return 0.0
        return float(np.max(self.run_times_sec)) / self.n_steps * 1000.0


def parse_run_times_from_file(path: Path) -> list[float]:
    """Extract all EnergyPlus Run Time values from a log file."""
    times: list[float] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in RUN_TIME_RE.finditer(text):
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        total_sec = hours * 3600 + minutes * 60 + seconds
        times.append(total_sec)
    return times


def collect_from_logs() -> dict[str, TimingStats]:
    """Parse known log files to collect per-building-type timing data."""
    stats: dict[str, TimingStats] = {}

    for filename, btype in LOG_FILE_TO_BUILDING_TYPE.items():
        path = LOGS_DIR / filename
        if not path.exists():
            log.warning("Log file not found: %s", path)
            continue

        run_times = parse_run_times_from_file(path)
        if not run_times:
            log.warning("No EnergyPlus Run Time entries in %s", path)
            continue

        if btype not in stats:
            stats[btype] = TimingStats(
                building_type=btype,
                source=filename,
            )

        stats[btype].run_times_sec.extend(run_times)
        log.info(
            "Parsed %d run times for %s from %s (mean=%.2fs)",
            len(run_times),
            btype,
            filename,
            float(np.mean(run_times)),
        )

    return stats


def run_benchmark_for_type(
    building_type: str,
    n_buildings: int = 3,
) -> TimingStats:
    """Run a live benchmark for a building type to measure step time."""
    import time

    from building2building.api import make_multizones_env, make_single_zone_env
    from building2building.types import RunPeriodConfig

    log.info(
        "Running benchmark for %s (%d buildings)...",
        building_type,
        n_buildings,
    )

    reward = {
        "reward_type": "DeadbandRewardConfig",
        "dT": 1.0,
        "energy_weight": 0.01,
    }
    task: dict[str, Any] = {"run_period": "full_year"}
    max_steps = RunPeriodConfig.from_name("full_year").expected_steps()

    run_times: list[float] = []

    for idx in range(n_buildings):
        eplus_dir = Path(tempfile.mkdtemp(prefix=f"bench_{building_type}_"))

        try:
            if building_type == "SingleZone":
                env = make_single_zone_env(
                    split="test",
                    split_index=idx,
                    eplus_output_dir=str(eplus_dir),
                    reward=reward,
                    task=task,
                )
            else:
                env = make_multizones_env(
                    building_type=building_type,
                    split="test",
                    split_index=idx,
                    eplus_output_dir=str(eplus_dir),
                    reward=reward,
                    task=task,
                )

            obs, _info = env.reset()
            done = False
            step = 0

            t0 = time.perf_counter()
            while not done and step < max_steps:
                action = env.action_space.sample()
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                step += 1
            elapsed = time.perf_counter() - t0

            run_times.append(elapsed)
            log.info(
                "  %s idx=%d: %d steps in %.2fs (%.3f ms/step)",
                building_type,
                idx,
                step,
                elapsed,
                elapsed / step * 1000 if step > 0 else 0,
            )

        except Exception:
            log.exception(
                "Failed benchmark for %s idx=%d", building_type, idx
            )
        finally:
            try:
                env.close()
            except Exception:
                pass

    return TimingStats(
        building_type=building_type,
        run_times_sec=run_times,
        n_steps=max_steps,
        source="live_benchmark",
    )


def format_latex_table(all_stats: dict[str, TimingStats]) -> str:
    """Format timing stats as a LaTeX table."""
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{EnergyPlus simulation step time per building type "
        r"(full-year rollout, 12 timesteps/hour, 10\,001 steps).}"
    )
    lines.append(r"\label{tab:step_time}")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"Building Type & Mean (ms/step) & Std (ms/step) & "
        r"Total (s) & $n$ \\"
    )
    lines.append(r"\midrule")

    for btype in BUILDING_TYPE_ORDER:
        if btype not in all_stats:
            display = BUILDING_TYPE_DISPLAY.get(btype, btype)
            lines.append(f"{display} & --- & --- & --- & 0 \\\\")
            continue

        s = all_stats[btype]
        display = BUILDING_TYPE_DISPLAY.get(btype, btype)
        lines.append(
            f"{display} & {s.mean_step_ms:.3f} & {s.std_step_ms:.3f} & "
            f"{s.mean_total_sec:.2f} & {s.n_samples} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def print_ascii_table(all_stats: dict[str, TimingStats]) -> None:
    """Print a readable ASCII table to stdout."""
    header = (
        f"{'Building Type':<25} {'Mean (ms/step)':>15} "
        f"{'Std (ms/step)':>14} {'Total (s)':>10} {'n':>5}"
    )
    sep = "=" * len(header)
    print()
    print(sep)
    print("EnergyPlus Simulation Step Time per Building Type")
    print("(full-year rollout, 12 timesteps/hour, 10,001 steps)")
    print(sep)
    print(header)
    print("-" * len(header))

    for btype in BUILDING_TYPE_ORDER:
        display = BUILDING_TYPE_DISPLAY.get(btype, btype)
        if btype not in all_stats:
            print(
                f"{display:<25} {'---':>15} {'---':>14} "
                f"{'---':>10} {'0':>5}"
            )
            continue

        s = all_stats[btype]
        print(
            f"{display:<25} {s.mean_step_ms:>15.3f} {s.std_step_ms:>14.3f} "
            f"{s.mean_total_sec:>10.2f} {s.n_samples:>5}"
        )

    print(sep)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="Run live benchmarks for building types with no log data.",
    )
    parser.add_argument(
        "--n-buildings",
        type=int,
        default=3,
        help="Number of buildings to benchmark per missing type.",
    )
    args = parser.parse_args()

    all_stats = collect_from_logs()

    missing = [bt for bt in BUILDING_TYPE_ORDER if bt not in all_stats]
    if missing:
        log.info("Missing timing data for: %s", missing)
        if args.run_missing:
            for btype in missing:
                stats = run_benchmark_for_type(
                    btype, n_buildings=args.n_buildings
                )
                if stats.n_samples > 0:
                    all_stats[btype] = stats
        else:
            log.info(
                "Use --run-missing to run live benchmarks for these types."
            )

    print_ascii_table(all_stats)

    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    latex = format_latex_table(all_stats)
    OUTPUT_TABLE_PATH.write_text(latex, encoding="utf-8")
    log.info("LaTeX table written to %s", OUTPUT_TABLE_PATH)


if __name__ == "__main__":
    main()
