from pathlib import Path

from building2building.pipeline import get_airflow_and_coil_node_setpoint_actuators


def test_get_airflow_and_coil_node_setpoint_actuators_finds_expected() -> None:
    edd_path = Path(__file__).parent / "fixtures" / "eplusout.edd"
    actuators = get_airflow_and_coil_node_setpoint_actuators(edd_path)

    keys = {
        (a["component_name"], a["component_type"], a["control_type"], a["units"])
        for a in actuators
    }

    # Fan air mass flow rate actuator
    assert (
        "MINI SPLIT HEAT PUMP SUPPLY FAN",
        "Fan",
        "Fan Air Mass Flow Rate",
        "[kg/s]",
    ) in keys

    # Coil node temperature setpoints (system node setpoint temperature setpoint)
    assert (
        "MINI SPLIT HEAT PUMP UNITARY SYSTEM FAN - COOLING COIL NODE",
        "System Node Setpoint",
        "Temperature Setpoint",
        "[C]",
    ) in keys
    assert (
        "MINI SPLIT HEAT PUMP UNITARY SYSTEM COOLING COIL - HEATING COIL NODE",
        "System Node Setpoint",
        "Temperature Setpoint",
        "[C]",
    ) in keys
    assert (
        "MINI SPLIT HEAT PUMP UNITARY SYSTEM HEATING COIL - SUPPLEMENTAL COIL NODE",
        "System Node Setpoint",
        "Temperature Setpoint",
        "[C]",
    ) in keys

    # Availability override for the air loop
    assert (
        "MINI SPLIT HEAT PUMP AIRLOOP",
        "AirLoopHVAC",
        "Availability Status",
        "[ ]",
    ) in keys


def test_get_airflow_and_coil_node_setpoint_actuators_includes_unitary_outlet_node() -> None:
    edd_path = Path(__file__).parent / "fixtures" / "eplusout.edd"
    actuators = get_airflow_and_coil_node_setpoint_actuators(
        edd_path, unitary_outlet_nodes=["NODE 3"]
    )

    keys = {
        (a["component_name"], a["component_type"], a["control_type"], a["units"])
        for a in actuators
    }

    assert ("NODE 3", "System Node Setpoint", "Temperature Setpoint", "[C]") in keys

