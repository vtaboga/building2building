from __future__ import annotations

import json
from pathlib import Path

from building2building.pipeline import add_node_setpoint_diagnostics_inplace


def test_add_node_setpoint_diagnostics_adds_expected_output_variables() -> None:
    src = Path("tests/fixtures/bldg1.epjson").resolve()
    assert src.exists()

    epjson = json.loads(src.read_text(encoding="utf-8"))
    add_node_setpoint_diagnostics_inplace(epjson, reporting_frequency="Timestep")

    outvars = epjson.get("Output:Variable", {})
    assert isinstance(outvars, dict) and outvars

    # For this fixture, the unitary system outlet node is "Node 3".
    wanted = {
        ("System Node Setpoint Temperature", "Node 3", "Timestep"),
        ("System Node Temperature", "Node 3", "Timestep"),
        ("System Node Mass Flow Rate", "Node 3", "Timestep"),
        ("Fan Air Mass Flow Rate", "mini split heat pump supply fan", "Timestep"),
    }

    found: set[tuple[str, str, str]] = set()
    for _k, obj_any in outvars.items():
        if not isinstance(obj_any, dict):
            continue
        vn = obj_any.get("variable_name")
        kv = obj_any.get("key_value")
        rf = obj_any.get("reporting_frequency")
        if isinstance(vn, str) and isinstance(kv, str) and isinstance(rf, str):
            found.add((vn, kv, rf))

    missing = wanted - found
    assert not missing, f"Missing Output:Variable entries: {missing}"

