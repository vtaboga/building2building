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

    # We expect a mix of:
    # - scheduled setpoints (Schedule:Constant / Schedule Value)
    # - direct fan airflow actuation (Fan / Fan Air Mass Flow Rate)
    for act in actuator_descriptions:
        assert act.units in ["Temperature", "Availability", "[kg/s]"]

        if act.component_type == "Schedule:Constant":
            assert act.control_type == "Schedule Value"
            assert act.component_name.startswith("B2B")
            assert act.units in ["Temperature", "Availability"]
        elif act.component_type == "Fan":
            assert act.control_type == "Fan Air Mass Flow Rate"
            assert act.units == "[kg/s]"
        else:
            raise AssertionError(
                f"Unexpected actuator type/control: {act.component_type} / {act.control_type}"
            )

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

    # Verify all schedules referenced by schedule actuators exist
    for act in actuator_descriptions:
        if act.component_type == "Schedule:Constant":
            assert act.component_name in epjson_data["Schedule:Constant"], (
                f"Schedule {act.component_name} should exist in epJSON"
            )

    print(f"\n✓ Control epJSON is valid")
    print(
        f"✓ Contains {len(epjson_data['Schedule:Constant'])} Schedule:Constant objects"
    )
    n_sched = sum(1 for a in actuator_descriptions if a.component_type == "Schedule:Constant")
    print(f"✓ All {n_sched} actuator schedules exist")
