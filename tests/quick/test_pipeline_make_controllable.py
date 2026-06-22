"""Pins the ``make_controllable`` actuator-emission contract per HVAC type.

Asserts that ``make_controllable`` produces the expected set of
``ActuatorDescription`` objects for each minimal building-type fixture (whose
HVAC archetype is VAV, Unitary, or HeatingOnly) and that every emitted actuator
appears in the resulting epJSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from building2building.env import STORE_PATH
from building2building.pipeline.actuators import make_controllable
from building2building.store import realize

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


_VAV_PAIRS = {("Schedule:Constant", "Schedule Value"), ("Outdoor Air Controller", "Air Mass Flow Rate")}
_UNITARY_PAIRS = {("Fan", "Fan Air Mass Flow Rate"), ("Schedule:Constant", "Schedule Value")}

# Map actuator component_type to the epJSON object-type key that should hold the
# component_name.  Only Schedule:Constant and Outdoor Air Controller are
# materialised by the pipeline as top-level epJSON sections whose keys we can
# check directly; Fan actuators reference fan objects whose type varies
# (Fan:SystemModel, Fan:OnOff, …) so we skip them here.
_COMPONENT_TYPE_TO_EPJSON_SECTION: dict[str, str] = {
    "Schedule:Constant": "Schedule:Constant",
    "Outdoor Air Controller": "Controller:OutdoorAir",
}


@pytest.mark.quick
@pytest.mark.parametrize(
    ("fixture_name", "required_pairs"),
    [
        ("minimal_officemedium", _VAV_PAIRS),
        ("minimal_officesmall", _UNITARY_PAIRS),
        ("minimal_restaurantfastfood", _UNITARY_PAIRS),
        ("minimal_singlefamilyhouse", _UNITARY_PAIRS),
        ("minimal_retailstandalone", _UNITARY_PAIRS),
        ("minimal_warehouse", _UNITARY_PAIRS),
    ],
)
def test_make_controllable_by_hvac_type(
    fixture_name: str, required_pairs: set[tuple[str, str]]
) -> None:
    epjson_path, equipment = realize(
        STORE_PATH.get(),
        make_controllable(FIXTURES_DIR / fixture_name / "building.epjson"),
    )
    epjson = json.loads(epjson_path.read_text())

    actuator_pairs = {
        (desc.component_type, desc.control_type)
        for eq in equipment
        for desc in eq.actuator_descriptions()
    }
    assert required_pairs.issubset(actuator_pairs)

    for component_type, control_type in actuator_pairs:
        assert "autosized" not in component_type.lower()
        assert "autosized" not in control_type.lower()

    # Every actuator whose component_type maps to a concrete epJSON section must
    # have a component_name that actually exists in that section.  This catches
    # the silent mismatch where emission code creates an actuator handle that
    # does not survive into the output epJSON (e.g. a renamed Schedule:Constant).
    all_descriptions = [
        desc
        for eq in equipment
        for desc in eq.actuator_descriptions()
    ]
    for desc in all_descriptions:
        section_key = _COMPONENT_TYPE_TO_EPJSON_SECTION.get(desc.component_type)
        if section_key is None:
            continue
        section = epjson.get(section_key, {})
        assert desc.component_name in section, (
            f"Actuator {desc.component_type!r} × {desc.control_type!r} "
            f"references component_name {desc.component_name!r} which is not "
            f"present in epjson[{section_key!r}] for fixture {fixture_name!r}. "
            f"Available keys: {sorted(section)[:10]}"
        )
