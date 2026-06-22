#!/usr/bin/env python3
"""Measure EnergyPlus per-cycle RSS growth for TODO B0.1.c.

Run this on a SLURM CPU node (not a login node) to produce the
authoritative per-cycle RSS figure that should be used to set
``_RSS_PER_CYCLE_BYTES`` in ``tests/long/test_env_leak.py``.

Usage::

    python scripts/diagnostics/measure_eplus_rss_growth.py \
        [--n 20] [--building-type SingleFamilyHouse]

The script prints:
  - per-cycle RSS growth for each of N cycles
  - mean and p95 across cycles
  - recommended ``_RSS_PER_CYCLE_BYTES`` value (p95 + 25% margin)

Update ``tests/long/test_env_leak.py::_RSS_PER_CYCLE_BYTES`` with the output.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Number of cycles")
    parser.add_argument(
        "--building-type",
        default="SingleFamilyHouse",
        help="Building type to use for the measurement",
    )
    args = parser.parse_args()

    try:
        import psutil
    except ImportError:
        print("ERROR: psutil is required. pip install psutil", file=sys.stderr)
        sys.exit(1)

    from building2building.env import setup_energyplus_path

    setup_energyplus_path()

    import building2building as b2b

    proc = psutil.Process()
    env_kwargs = dict(
        building_type=args.building_type,
        split="train",
        index=0,
        task="task1",
        run_period="winter",
        timesteps_per_hour=4,
    )

    # Warm up: one cycle outside measurement to let EnergyPlus initialise
    # any one-time global state.
    print("Warming up (1 cycle outside measurement)...", flush=True)
    env = b2b.make_env(**env_kwargs)
    env.reset()
    env.close()
    gc.collect()

    print(f"Measuring {args.n} cycles...", flush=True)
    growths: list[int] = []
    for i in range(args.n):
        gc.collect()
        rss_before = proc.memory_info().rss

        env = b2b.make_env(**env_kwargs)
        env.reset()
        env.close()

        gc.collect()
        rss_after = proc.memory_info().rss
        delta = rss_after - rss_before
        growths.append(delta)
        print(f"  cycle {i:>3d}: {delta / 1e6:+.1f} MB", flush=True)

    import statistics

    mean_mb = statistics.mean(growths) / 1e6
    p95_bytes = sorted(growths)[int(0.95 * len(growths))]
    p95_mb = p95_bytes / 1e6
    recommended_bytes = int(p95_bytes * 1.25)
    recommended_mb = recommended_bytes / 1e6

    print()
    print(f"Results ({args.n} cycles, building_type={args.building_type!r}):")
    print(f"  mean:       {mean_mb:+.1f} MB/cycle")
    print(f"  p95:        {p95_mb:+.1f} MB/cycle")
    print(
        f"  recommended _RSS_PER_CYCLE_BYTES = {recommended_bytes}  "
        f"  # {recommended_mb:.0f} MB/cycle (p95 + 25%)"
    )
    print()
    print("Update tests/long/test_env_leak.py:")
    print(
        f"  _RSS_PER_CYCLE_BYTES = {recommended_bytes}"
        f"  # {recommended_mb:.0f} MB/cycle (p95 + 25% margin, measured)"
    )
    print()
    print("Measured residual EnergyPlus-native RSS growth:")
    print(
        f"  mean {mean_mb:.0f} MB/cycle, p95 {p95_mb:.0f} MB/cycle "
        f"({args.building_type})"
    )


if __name__ == "__main__":
    main()
