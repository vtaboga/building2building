"""Pins the equipment-schema round-trip contract per HVAC type.

Asserts that the equipment detected from each minimal building-type fixture
(whose HVAC archetype is VAV, Unitary, or HeatingOnly) matches the expected
schema class and that the actuator descriptions survive the
structure → unstructure → structure round-trip without loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cattrs import structure, unstructure

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

    # Every required type must appear at least once (not just be a permitted type).
    for required_type in required_types:
        assert any(isinstance(item, required_type) for item in equipment), (
            f"Expected at least one {required_type.__name__} in {fixture_name}/equipment.json, "
            f"got types: {[type(item).__name__ for item in equipment]}"
        )

    # structure → unstructure → structure round-trip: actuator descriptions must
    # survive serialization without loss.
    roundtripped = structure(unstructure(equipment), list[AnyEquipment])
    assert len(roundtripped) == len(equipment)
    for original, restored in zip(equipment, roundtripped):
        assert type(original) is type(restored)
        orig_descs = original.actuator_descriptions()
        rest_descs = restored.actuator_descriptions()
        assert len(orig_descs) == len(rest_descs)
        for od, rd in zip(orig_descs, rest_descs):
            assert od == rd, (
                f"Actuator description changed after round-trip in {fixture_name}: "
                f"{od!r} → {rd!r}"
            )
