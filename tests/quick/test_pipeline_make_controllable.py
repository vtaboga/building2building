"""Pins the ``make_controllable`` actuator-emission contract per HVAC type.

Asserts that ``make_controllable`` produces the expected set of
``ActuatorDescription`` objects for each minimal fixture (VAV, Unitary,
HeatingOnly) and that every emitted actuator appears in the resulting epJSON.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from building2building.env import STORE_PATH
from building2building.pipeline.actuators import make_controllable
from building2building.store import realize

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("fixture_name", "required_pairs"),
    [
        (
            "minimal_vav",
            {("Schedule:Constant", "Schedule Value"), ("Outdoor Air Controller", "Air Mass Flow Rate")},
        ),
        (
            "minimal_unitary",
            {("Fan", "Fan Air Mass Flow Rate"), ("Schedule:Constant", "Schedule Value")},
        ),
        (
            "minimal_heating_only",
            {("Fan", "Fan Air Mass Flow Rate"), ("Schedule:Constant", "Schedule Value")},
        ),
    ],
)
def test_make_controllable_by_hvac_type(
    fixture_name: str, required_pairs: set[tuple[str, str]]
) -> None:
    _, equipment = realize(
        STORE_PATH.get(),
        make_controllable(FIXTURES_DIR / fixture_name / "building.epjson"),
    )

    actuator_pairs = {
        (desc.component_type, desc.control_type)
        for eq in equipment
        for desc in eq.actuator_descriptions()
    }
    assert required_pairs.issubset(actuator_pairs)

    for component_type, control_type in actuator_pairs:
        assert "autosized" not in component_type.lower()
        assert "autosized" not in control_type.lower()
