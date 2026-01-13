from __future__ import annotations

import json
import tempfile
from pathlib import Path

from building2building.pipeline import set_unitary_systems_to_setpoint_control_inplace


def test_pipeline_sets_unitary_systems_to_setpoint_control() -> None:
    src = Path("tests/fixtures/bldg1.epjson").resolve()
    assert src.exists()

    data = json.loads(src.read_text(encoding="utf-8"))
    set_unitary_systems_to_setpoint_control_inplace(data)

    unitary = data.get("AirLoopHVAC:UnitarySystem", {})
    assert isinstance(unitary, dict) and unitary, "Fixture should contain a unitary system"
    for _name, obj in unitary.items():
        assert isinstance(obj, dict)
        assert obj.get("control_type") == "SetPoint"
        sched = obj.get("supply_air_fan_operating_mode_schedule_name")
        assert isinstance(sched, str) and sched.strip()

    sched_const = data.get("Schedule:Constant", {})
    assert isinstance(sched_const, dict) and sched_const, "Expected a constant schedule to be created"

