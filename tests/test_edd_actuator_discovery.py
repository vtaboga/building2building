"""Test that make_controllable creates expected actuators for the fixture building."""

from pathlib import Path

from building2building.env import STORE_PATH
from building2building.pipeline import make_controllable
from building2building.store import Constant, realize


def test_make_controllable_creates_expected_actuators() -> None:
    """Test that make_controllable creates controllable schedules for HVAC systems."""
    epjson_path = Path(__file__).parent / "fixtures" / "bldg1.epjson"
    assert epjson_path.exists()

    # Make the building controllable
    control_expr = make_controllable(Constant(epjson_path))
    control_epjson, actuator_descriptions = realize(STORE_PATH.get(), control_expr)

    # Should have created actuators
    assert len(actuator_descriptions) > 0, "Should create at least one actuator"

    # All actuators should be Schedule:Constant / Schedule Value
    for act in actuator_descriptions:
        assert act.component_type == "Schedule:Constant"
        assert act.control_type == "Schedule Value"
        assert act.component_name.startswith("B2B")
        assert act.units in ["Temperature", "Availability"]

    # Check that we have actuators for the mini split heat pump system
    # The fixture has a unitary system, so we should have temperature setpoint schedules
    temp_actuators = [a for a in actuator_descriptions if a.units == "Temperature"]
    assert len(temp_actuators) > 0, (
        "Should have temperature setpoint actuators for unitary system"
    )

    print(f"\n✓ Created {len(actuator_descriptions)} actuators:")
    for act in actuator_descriptions:
        print(f"  - {act.component_name} ({act.units})")


def test_make_controllable_produces_valid_epjson() -> None:
    """Test that make_controllable produces a valid epJSON file."""
    import json

    epjson_path = Path(__file__).parent / "fixtures" / "bldg1.epjson"
    assert epjson_path.exists()

    # Make the building controllable
    control_expr = make_controllable(Constant(epjson_path))
    control_epjson, actuator_descriptions = realize(STORE_PATH.get(), control_expr)

    # Verify the output epJSON is valid
    assert control_epjson.exists()

    with open(control_epjson) as f:
        epjson_data = json.load(f)

    # Should have Schedule:Constant objects
    assert "Schedule:Constant" in epjson_data
    assert len(epjson_data["Schedule:Constant"]) > 0

    # Should have ScheduleTypeLimits
    assert "ScheduleTypeLimits" in epjson_data

    # Verify all schedules referenced by actuators exist
    for act in actuator_descriptions:
        assert act.component_name in epjson_data["Schedule:Constant"], (
            f"Schedule {act.component_name} should exist in epJSON"
        )

    print(f"\n✓ Control epJSON is valid")
    print(
        f"✓ Contains {len(epjson_data['Schedule:Constant'])} Schedule:Constant objects"
    )
    print(f"✓ All {len(actuator_descriptions)} actuator schedules exist")
