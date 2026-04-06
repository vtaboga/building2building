"""Benchmark orchestration for evaluating RL policies across buildings.

Exposes benchmark problem classes that define train/test splits over the
building dataset:

* :class:`SingleTypeTrainTestBenchmark` -- within a single building type.
* :class:`MultiTypeTrainTestBenchmark` -- across different building types.
* :class:`AdaptiveDynamicsProblem` -- parametric dynamics variation.
"""

from __future__ import annotations

from building2building.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem
from building2building.benchmark.problem_multizones_splits import (
    MultiTypeTrainTestBenchmark,
    SingleTypeTrainTestBenchmark,
)

__all__ = [
    "AdaptiveDynamicsProblem",
    "SingleTypeTrainTestBenchmark",
    "MultiTypeTrainTestBenchmark",
]
