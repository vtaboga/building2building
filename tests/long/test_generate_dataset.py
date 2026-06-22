"""Stage 2 smoke test: generate_dataset.py round-trip for one OfficeMedium building.

Skipped unless B2B_RUN_LONG_TESTS=1 and HF credentials allow downloading
vtaboga/building2building_dataset splits.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.long


def _requires_long_runtime() -> None:
    if os.environ.get("B2B_RUN_LONG_TESTS", "0") != "1":
        pytest.skip("Set B2B_RUN_LONG_TESTS=1 to run long tests")


def test_generate_dataset_officemedium_smoke(tmp_path: Path) -> None:
    """Regenerate 1 OfficeMedium building; validate the written artefacts.

    Acceptance criteria (for the agent-facing action_dim contract):
    - equipment.json round-trips through cattrs.structure, and the raw
      actuator count matches the generation summary.
    - action_dim in the rewritten metadata.parquet equals the *agent-facing*
      dimension ``agent_action_dim(equipment_list)`` (fixed actuators are
      filtered, so this is <= the raw actuator count) and reproduces the
      value published in the upstream metadata.parquet.
    - splits.json is bit-identical to the HF upstream copy.
    """
    _requires_long_runtime()

    import shutil

    import cattrs
    import pandas as pd

    from building2building.data.download import download_splits
    from building2building.env import setup_energyplus_path
    from building2building.pipeline.generate_dataset import (
        _load_splits,
        generate_one_building,
        rebuild_metadata_parquet,
        copy_splits,
    )
    from building2building.pipeline.actuators import (
        VAVSystem,
        UnitarySystem,
        HeatingOnlyZone,
    )

    setup_energyplus_path()

    building_type = "OfficeMedium"
    out_root = tmp_path / "staging"
    out_root.mkdir()

    ids = _load_splits([building_type])
    assert building_type in ids, f"No IDs for {building_type} in splits.json"

    # Take the first ID only.
    processed_id = ids[building_type][0]
    summary = generate_one_building(
        building_type=building_type,
        processed_id=processed_id,
        out_root=out_root,
    )
    bldg_dir = out_root / building_type / processed_id

    # All three artefact files must exist.
    for fname in ("building.epjson", "equipment.json", "metadata.json"):
        assert (bldg_dir / fname).exists(), f"{fname} not found in {bldg_dir}"

    # equipment.json must round-trip through cattrs.
    raw_eq = json.loads((bldg_dir / "equipment.json").read_text())
    assert isinstance(raw_eq, list) and len(raw_eq) > 0

    n_actuators_from_eq = 0
    for entry in raw_eq:
        et = entry.get("equipment_type", "")
        if et == "vavsystem":
            obj = cattrs.structure(entry, VAVSystem)
        elif et == "unitarysystem":
            obj = cattrs.structure(entry, UnitarySystem)
        elif et == "heatingonlyzone":
            obj = cattrs.structure(entry, HeatingOnlyZone)
        else:
            pytest.fail(f"Unknown equipment_type {et!r} in equipment.json")
        n_actuators_from_eq += len(obj.actuator_descriptions())

    assert n_actuators_from_eq == summary["num_actuators"], (
        f"actuator count mismatch: summary says {summary['num_actuators']}, "
        f"re-counted {n_actuators_from_eq} from equipment.json"
    )

    # Rebuild metadata.parquet and verify the AGENT-FACING action_dim.
    #
    # rebuild_metadata_parquet is, by design, a whole-building-type operation:
    # it loads the published metadata and requires a regenerated artefact dir
    # for *every* building of the type (fail-loud, no silent skip).  This smoke
    # test regenerates a single building, so we scope the function to that one
    # building by pointing its metadata source at a single-row stand-in built
    # from the published parquet.  This exercises the real rewrite path
    # (equipment structuring + agent_action_dim + parquet write) without
    # weakening the production "require every building" guard.
    from unittest import mock

    from building2building.data.download import download_metadata
    from building2building.pipeline.actuators import AnyEquipment
    from building2building.simulator.action_spaces import agent_action_dim

    published = pd.read_parquet(download_metadata())
    one_row = published[published["building_id"] == processed_id].copy()
    assert len(one_row) == 1, f"{processed_id!r} not in published metadata.parquet"
    published_action_dim = int(one_row.iloc[0]["action_dim"])

    stand_in = tmp_path / "metadata_one_row.parquet"
    one_row.to_parquet(stand_in, index=False)

    with mock.patch(
        "building2building.pipeline.generate_dataset.download_metadata",
        return_value=stand_in,
    ):
        rebuild_metadata_parquet(out_root, [building_type])

    parquet_path = out_root / "metadata.parquet"
    assert parquet_path.exists(), "metadata.parquet not created"

    df = pd.read_parquet(parquet_path)
    row = df[df["building_id"] == processed_id]
    assert (
        len(row) == 1
    ), f"processed_id={processed_id!r} not found in rewritten metadata.parquet"

    # action_dim is the AGENT-FACING action-space dimension (G5): agent_action_dim
    # filters the fixed (non-agent) actuators, so it is <= the raw actuator count.
    equipment_list = cattrs.structure(raw_eq, list[AnyEquipment])
    expected_agent_dim = agent_action_dim(equipment_list)
    assert int(row.iloc[0]["action_dim"]) == expected_agent_dim, (
        f"rewritten action_dim ({row.iloc[0]['action_dim']}) "
        f"!= agent_action_dim ({expected_agent_dim})"
    )
    assert 0 < expected_agent_dim <= n_actuators_from_eq, (
        f"agent-facing dim {expected_agent_dim} must be in "
        f"(0, raw actuator count {n_actuators_from_eq}]"
    )
    assert expected_agent_dim == published_action_dim, (
        f"regenerated agent_action_dim ({expected_agent_dim}) "
        f"!= published metadata action_dim ({published_action_dim})"
    )

    # Copy splits.json and verify it matches the HF upstream.
    copy_splits(out_root)
    staging_splits = json.loads((out_root / "splits.json").read_text())
    upstream_splits = json.loads(download_splits().read_text())
    assert (
        staging_splits == upstream_splits
    ), "splits.json written by copy_splits differs from the HF upstream"
