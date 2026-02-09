from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from b2b.make_env import make_env
from b2b.utils import HydroQuebecRowIdSplits


@dataclass(frozen=True)
class ActionSpaceSelectionRecord:
    split: str
    split_index: int
    building_id: int | None
    dataset_row_index: int | None
    n_conditioned_zones: int
    n_unconditioned_zones: int
    actuator_names: list[str]
    success: bool
    error: str | None


def _is_fan_actuator(name: str) -> bool:
    s = name.lower()
    return s.startswith("fan::") and "fan air mass flow rate" in s


def _is_node_actuator(name: str) -> bool:
    s = name.lower()
    if s.startswith("schedule:constant::schedule value::"):
        return True
    if s.startswith("system node setpoint::temperature setpoint::"):
        return True
    return False


def _parse_building_id_from_action_names(action_names: list[str]) -> int | None:
    """
    Best-effort parsing: sometimes actuator component_name includes a schedule name
    but not an ID. We keep this optional and do not depend on it for correctness.
    """
    _ = action_names
    return None


def test_hydroquebec_selection_action_space_has_one_fan_one_node(tmp_path: Path) -> None:
    """
    Integration test: iterate over the train/test split ids and ensure each
    selected building yields an action space with exactly 2 actuators:
    - 1 fan actuator
    - 1 node temperature setpoint actuator

    Note: this is slow (downloads + processing).
    """

    # Optional safety valve for debugging.
    limit_raw = os.environ.get("B2B_HQ_SELECTION_VALIDATION_LIMIT", "").strip()
    limit = int(limit_raw) if limit_raw else 0

    splits = HydroQuebecRowIdSplits.load_from_action_space_2_zone_1()
    split_map: dict[str, list[int]] = {"train": splits.train_row_ids, "test": splits.test_row_ids}

    records: list[ActionSpaceSelectionRecord] = []
    failures: list[ActionSpaceSelectionRecord] = []

    for split, ids in split_map.items():
        for split_index in range(len(ids)):
            if limit and len(records) >= limit:
                break

            cfg = OmegaConf.create(
                {
                    # Keep config minimal; only bldg/reward/env are read by search_configs().
                    "env": {"control_mode": "hvac_actuators", "max_steps": 1, "normalize_obs": False},
                    "reward": {"reward_type": None},
                    "bldg": {
                        "selection": {"enabled": True, "split": split, "index": split_index},
                        "bldg": {},
                    },
                }
            )

            env = None
            try:
                env = make_env(config=cfg, eplus_output_dir=str(tmp_path / "eplus_outputs"))
                meta = getattr(env, "metadata", {}) or {}

                action_names_any = meta.get("action_names", [])
                action_names = (
                    [str(x) for x in action_names_any]
                    if isinstance(action_names_any, list)
                    else []
                )

                controlled_zones_any = meta.get("controlled_zones", [])
                controlled_zones = (
                    [str(x) for x in controlled_zones_any]
                    if isinstance(controlled_zones_any, list)
                    else []
                )
                uncontrolled_zones_any = meta.get("uncontrolled_zones", [])
                uncontrolled_zones = (
                    [str(x) for x in uncontrolled_zones_any]
                    if isinstance(uncontrolled_zones_any, list)
                    else []
                )

                building_source = meta.get("building_source_metadata", {})
                dataset_row_index = None
                if isinstance(building_source, dict):
                    v = building_source.get("dataset_row_index")
                    if isinstance(v, int):
                        dataset_row_index = v

                # Primary check: action space corresponds to exactly two actuators.
                if not hasattr(env, "action_space") or not hasattr(env.action_space, "shape"):
                    raise AssertionError("env.action_space.shape missing")
                if tuple(getattr(env.action_space, "shape")) != (2,):
                    raise AssertionError(f"Expected action_space.shape==(2,), got {env.action_space.shape!r}")

                if len(action_names) != 2:
                    raise AssertionError(f"Expected exactly 2 action_names, got {len(action_names)}: {action_names}")

                fan_count = sum(1 for n in action_names if _is_fan_actuator(n))
                node_count = sum(1 for n in action_names if _is_node_actuator(n))
                if fan_count != 1 or node_count != 1:
                    raise AssertionError(
                        "Expected exactly one fan and one node actuator. "
                        f"fan_count={fan_count} node_count={node_count} action_names={action_names}"
                    )

                rec = ActionSpaceSelectionRecord(
                    split=split,
                    split_index=split_index,
                    building_id=_parse_building_id_from_action_names(action_names),
                    dataset_row_index=dataset_row_index,
                    n_conditioned_zones=len(controlled_zones),
                    n_unconditioned_zones=len(uncontrolled_zones),
                    actuator_names=action_names,
                    success=True,
                    error=None,
                )
                records.append(rec)
            except Exception as e:
                rec = ActionSpaceSelectionRecord(
                    split=split,
                    split_index=split_index,
                    building_id=None,
                    dataset_row_index=None,
                    n_conditioned_zones=0,
                    n_unconditioned_zones=0,
                    actuator_names=[],
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                )
                records.append(rec)
                failures.append(rec)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

    out_path = tmp_path / "hq_action_space_selection_records.json"
    out_path.write_text(json.dumps([r.__dict__ for r in records], indent=2) + "\n", encoding="utf-8")

    if failures:
        # Keep the failure message short; details are in the JSON record file.
        first = failures[0]
        pytest.fail(
            f"HydroQuebec selection validation failed for {len(failures)}/{len(records)} buildings. "
            f"First failure: split={first.split} index={first.split_index} error={first.error}. "
            f"Records written to: {out_path}"
        )

