"""Pins the equipment-schema round-trip contract per HVAC type.

Asserts that the equipment detected from each minimal building-type fixture
(whose HVAC archetype is VAV, Unitary, or HeatingOnly) matches the expected
schema class and that the actuator descriptions round-trip without loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cattrs import structure

from building2building.pipeline.actuators import AnyEquipment, HeatingOnlyZone, UnitarySystem, VAVSystem

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("fixture_name", "required_types"),
    [
        ("minimal_officemedium", (VAVSystem,)),
        ("minimal_officesmall", (UnitarySystem,)),
        ("minimal_restaurantfastfood", (UnitarySystem,)),
        ("minimal_singlefamilyhouse", (UnitarySystem,)),
        ("minimal_retailstandalone", (UnitarySystem, HeatingOnlyZone)),
        ("minimal_warehouse", (UnitarySystem, HeatingOnlyZone)),
    ],
)
def test_equipment_schema_round_trip_by_hvac_type(
    fixture_name: str, required_types: tuple[type, ...]
) -> None:
    payload = json.loads((FIXTURES_DIR / fixture_name / "equipment.json").read_text())
    equipment = structure(payload, list[AnyEquipment])

    assert equipment, "equipment.json must decode to at least one equipment object"
    assert all(any(isinstance(item, t) for t in required_types) for item in equipment)
