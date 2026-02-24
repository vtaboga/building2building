from __future__ import annotations

from typing import Any

from b2b.benchmark.problem_multizones_splits import build_interface_from_cfg


def build_multizones_split_benchmark(config: dict[str, Any]):
    return build_interface_from_cfg(config)


__all__ = ["build_multizones_split_benchmark"]
