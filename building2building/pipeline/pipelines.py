from building2building.pipeline.hvac.unitary import (
    add_node_setpoint_diagnostics,
    ensure_scheduled_node_temperature_setpoints,
    set_unitary_systems_to_setpoint_control,
)
from building2building.pipeline.steps.conversion import convert_idf, upgrade
from building2building.pipeline.steps.outputs import (
    add_edd_output,
    add_hvac_meters,
    add_outdoor_air_meters,
    add_tabular_output,
    modify_timestep,
)
from building2building.store import Derivation, Realizable, Rename


def create_discovery_pipeline(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
) -> Derivation:
    """
    Create an epJSON suitable for *dummy simulations* whose purpose is to generate
    discovery artifacts such as `eplusout.edd` (actuator dictionary) and
    `eplustbl.htm` (tabular summary).

    Critical behavior:
    - Keep the building's default thermostat / load-based HVAC control intact.
      Some models rely on that to run successfully and to expose expected
      actuators in `.edd`.

    Downstream, you can convert this into a control-ready epJSON for RL by
    calling `create_control_pipeline(discovery_epjson)`.
    """
    current = upgrade(input_file, energyplus_path, src_version)
    current = convert_idf(current, energyplus_path)

    # Add meters and monitoring used by downstream parsers / diagnostics
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)
    current = add_edd_output(current)
    current = add_tabular_output(current)

    # Configure simulation output resolution
    current = modify_timestep(current, timesteps_per_hour=4)

    # IMPORTANT: include scheduled node setpoints (SetpointManager:Scheduled + Schedule:Constant)
    # already in the discovery model so the dummy simulation's `.edd` contains the
    # corresponding `Schedule:* / Schedule Value` actuators.
    #
    # This does NOT change HVAC control mode (we keep the building's thermostat/load control),
    # but it makes setpoint control robustly actuable downstream.
    current = ensure_scheduled_node_temperature_setpoints(current, temperature_c=22.0)

    # Make the name useful and explicit.
    current = Rename("building.discovery.epjson", current)
    return current


def create_control_pipeline(discovery_epjson: Derivation) -> Derivation:
    """
    Convert a discovery epJSON into a control-ready epJSON for the "airflow +
    System Node Setpoint" control mode.

    This is intentionally applied *after* discovery simulations, because those
    simulations rely on default thermostat/load control, while our RL control
    mode relies on `AirLoopHVAC:UnitarySystem.control_type = SetPoint`.
    """
    current = set_unitary_systems_to_setpoint_control(discovery_epjson)
    # Provide scheduled setpoints required by EnergyPlus for SetPoint control.
    # Implemented via SetpointManager:Scheduled + Schedule:Constant so control can
    # robustly override the schedule value at runtime.
    current = ensure_scheduled_node_temperature_setpoints(current, temperature_c=22.0)
    # Output variables to let us verify node temperatures + setpoints in `eplusout.csv/sql`.
    current = add_node_setpoint_diagnostics(current)
    # Ensure `.edd` generation is enabled (idempotent) for subsequent runs.
    current = add_edd_output(current)
    current = Rename("building.epjson", current)
    return current

