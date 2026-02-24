#!/usr/bin/env python3
"""Example benchmark interface for multizones split selection.

This script demonstrates two interfaces:
1) single-type train/test split benchmark
2) multi-type train/test split benchmark

By default it only selects building IDs and writes a summary JSON.
Set `output.resolve_configs=true` to resolve full BuildingConfig objects.

Example (OfficeMedium actuator-shift benchmark):
    python scripts/benchmark_multizones_splits.py \\
        --config-name benchmark_multizones_officemedium_actuator_shift

Reverse the train/test actuator access:
    python scripts/benchmark_multizones_splits.py \\
        --config-name benchmark_multizones_officemedium_actuator_shift \\
        benchmark_interface.train.config.actuator_access.include_zone_heating_setpoints=true \\
        benchmark_interface.test.config.actuator_access.include_zone_heating_setpoints=false
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from b2b.benchmarks import build_multizones_split_benchmark
from b2b.benchmark.problem_multizones_splits import (
    MultiTypeTrainTestBenchmark,
    SingleTypeTrainTestBenchmark,
)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="benchmark_multizones_splits",
)
def main(cfg: DictConfig) -> int:
    output_dir = Path.cwd()
    cfg_dict_any = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = cfg_dict_any if isinstance(cfg_dict_any, dict) else {}
    iface = build_multizones_split_benchmark(cfg_dict)
    train_ids, test_ids = iface.select_building_ids()

    summary: dict[str, Any] = {
        "interface": type(iface).__name__,
        "train_building_ids": train_ids,
        "test_building_ids": test_ids,
    }

    if isinstance(iface, SingleTypeTrainTestBenchmark):
        summary["building_type"] = iface.building_type
    if isinstance(iface, MultiTypeTrainTestBenchmark):
        summary["train_types"] = list(iface.train_types)
        summary["test_types"] = list(iface.test_types)

    resolve_configs = bool(cfg.output.resolve_configs)
    if resolve_configs:
        result = iface.build_configs(eplus_output_dir=output_dir / "eplus_outputs")
        summary["n_train_configs"] = len(result.train_configs)
        summary["n_test_configs"] = len(result.test_configs)

    summary["config"] = OmegaConf.to_container(cfg, resolve=True)
    summary_path = output_dir / str(cfg.output.summary_file)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
