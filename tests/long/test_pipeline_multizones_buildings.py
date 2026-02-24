from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from b2b.benchmark.rollout_multizones import ALL_BUILDING_TYPES, run_multizones_rollout


pytestmark = pytest.mark.long


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long simulation tests")


@pytest.mark.parametrize("building_type", ALL_BUILDING_TYPES)
def test_multizones_pipeline_baseline_execution_per_type(
    building_type: str,
    tmp_path: Path,
) -> None:
    _requires_long_runtime()
    cfg = OmegaConf.create(
        {
            "policy": {"type": "air_loop_sat" if building_type == "OfficeMedium" else "unitary_sat"},
            "multizones": {"types": [building_type], "n_per_type": 1},
            "env": {"max_steps": 8},
            "task": {"run_period": "winter"},
            "reward": {"reward_type": "BaseRewardConfig"},
        }
    )
    records = run_multizones_rollout(cfg, output_dir=tmp_path / "rollout")
    assert len(records) == 1
    assert records[0].building_type == building_type
    assert records[0].success is True
