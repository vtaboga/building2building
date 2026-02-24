"""Test that make_controllable creates expected actuators for the fixture building."""

from pathlib import Path

from b2b.env import STORE_PATH
from b2b.pipeline import make_controllable
from b2b.pipeline.actuators import (
    make_vav_reheat_controllable,
    make_zone_temperature_control_controllable,
)
from b2b.simulator.controllable_zones import (
    get_vav_reheat_controllable_zones,
    get_zone_temperature_control_controllable_zones,
)
from b2b.store import Constant, realize


def test_zone_temperature_control_and_vav_reheat_actuators() -> None:
    """Test zone_temperature_control and vav_reheat produce expected actuators."""
    epjson = {
        "Zone": {"Core_bottom": {}, "Core_mid": {}},
        "ZoneControl:Thermostat": {
            "Core_bottom Thermostat": {"zone_or_zonelist_name": "Core_bottom"},
            "Core_mid Thermostat": {"zone_or_zonelist_name": "Core_mid"},
        },
        "AirTerminal:SingleDuct:VAV:Reheat": {
            "Core_bottom VAV Box": {},
            "Core_mid VAV Box": {},
        },
    }
    _, ztc = make_zone_temperature_control_controllable(epjson)
    _, vav = make_vav_reheat_controllable(epjson)

    assert len(ztc) == 4  # 2 zones x 2 setpoints
    assert all(a.component_type == "Zone Temperature Control" for a in ztc)
    assert {a.control_type for a in ztc} == {"Heating Setpoint", "Cooling Setpoint"}
    assert all(a.units == "[C]" for a in ztc)

    assert len(vav) == 2
    assert all(a.component_type == "AirTerminal:SingleDuct:VAV:Reheat" for a in vav)
    assert all(a.control_type == "Primary Air Maximum Flow Fraction" for a in vav)
    assert all(a.units == "[ ]" for a in vav)

    ztc_zones = get_zone_temperature_control_controllable_zones(epjson)
    assert "Core_bottom" in ztc_zones and "Core_mid" in ztc_zones


def test_make_controllable_creates_expected_actuators() -> None:
    """Test that make_controllable creates controllable schedules for HVAC systems."""
    epjson_path = Path(__file__).parent / "fixtures" / "bldg1.epjson"
    assert epjson_path.exists()

    # Make the building controllable
    control_expr = make_controllable(Constant(epjson_path))
    control_epjson, actuator_descriptions = realize(STORE_PATH.get(), control_expr)

    # Should have created actuators
    assert len(actuator_descriptions) > 0, "Should create at least one actuator"

    # We expect a mix of actuator types from make_controllable:
    # - Schedule:Constant / Schedule Value (setpoints, availability)
    # - Fan / Fan Air Mass Flow Rate
    # - Zone Temperature Control (Heating/Cooling Setpoint)
    # - AirTerminal:SingleDuct:VAV:Reheat (Primary Air Maximum Flow Fraction)
    # - WaterHeater / Setpoint Temperature, etc.
    allowed_units = {"Temperature", "Availability", "[kg/s]", "[C]", "[ ]"}
    for act in actuator_descriptions:
        assert act.units in allowed_units, f"Unexpected units: {act.units}"

        if act.component_type == "Schedule:Constant":
            assert act.control_type == "Schedule Value"
            assert act.component_name.startswith("B2B")
            assert act.units in ["Temperature", "Availability"]
        elif act.component_type == "Fan":
            assert act.control_type == "Fan Air Mass Flow Rate"
            assert act.units == "[kg/s]"
        elif act.component_type == "Zone Temperature Control":
            assert act.control_type in ("Heating Setpoint", "Cooling Setpoint")
            assert act.units == "[C]"
        elif act.component_type == "AirTerminal:SingleDuct:VAV:Reheat":
            assert act.control_type == "Primary Air Maximum Flow Fraction"
            assert act.units == "[ ]"
        elif act.component_type == "WaterHeater":
            assert act.control_type == "Setpoint Temperature"
            assert act.units == "Temperature"
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
