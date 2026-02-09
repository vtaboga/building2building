"""
Pipeline package.
"""

from pathlib import Path

from b2b.pipeline.actuators import ActuatorDescription, make_controllable
from b2b.pipeline.discovery import Metadata, extract_discovery_metadata
from b2b.pipeline.parse_edd import (
    EddActuatorDescriptor,
    get_airflow_and_coil_node_setpoint_actuators,
    get_b2b_scheduled_node_setpoint_actuators,
    get_hvac_actuators,
    get_schedule_value_actuators,
    get_zone_temperature_control_actuators,
    iter_edd_actuators,
)
from b2b.pipeline.parse_reports import (
    get_net_conditioned_area,
    get_warmup_days,
)
from b2b.pipeline.simulation import (
    detect_warmup_phases,
    eddfile,
    eiofile,
    eplustbl,
    run_simulation,
)
from b2b.pipeline.steps.conversion import (
    ConvertIDF,
    Transition,
    all_transitions,
    convert_idf,
    scan_upgraders,
    upgrade,
    upgrade_idf,
)
from b2b.pipeline.steps.outputs import (
    add_all_outputs,
    add_edd_output,
    add_hvac_meters,
    add_outdoor_air_meters,
    add_sqlite_output,
    add_tabular_output,
    modify_timestep,
    modify_run_period,
)
from b2b.pipeline.steps.schedule_files import link_in_schedule
from b2b.pipeline.steps.surfaces import GlueSurfaces, glue_surfaces
from b2b.pipeline.steps.thermostat_setpoints import (
    AddSetpointControl,
    add_setpoint_control,
    get_temperature_setpoints,
)
from b2b.store import Derivation, Expression, Realizable, Rename


def prepare_building(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
) -> Derivation:
    """
    Convert IDF to epJSON and prepare for controllability.

    Steps:
    1. Upgrade IDF to target EnergyPlus version
    2. Convert IDF → epJSON
    3. Add HVAC meters (electricity, gas) - needed for RL reward calculation
    4. Add outdoor air meters - needed for RL observations
    5. Modify timestep to 4 steps/hour

    Does NOT add discovery outputs (EDD, tabular, SQLite) or modify HVAC control.
    Use make_controllable() and extract_discovery_metadata() for those.

    Returns:
        Derivation resolving to prepared epJSON
    """
    current = upgrade(input_file, energyplus_path, src_version)
    current = convert_idf(current, energyplus_path)
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)
    current = modify_timestep(current, timesteps_per_hour=4)
    current = modify_run_period(current, begin_day_of_month=1, begin_month=1, end_day_of_month=31, end_month=12)
    current = Rename("building.epjson", current)
    return current


def create_complete_pipeline(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
) -> Expression[tuple[Path, list[ActuatorDescription]]]:
    """
    Complete pipeline: IDF → controllable epJSON with actuators.

    Standard pipeline for converting IDF files to control-ready epJSON:
    1. Upgrade and convert IDF
    2. Add meters needed for RL (HVAC energy, outdoor air)
    3. Configure timestep
    4. Make HVAC systems controllable

    For metadata extraction (area, warmup_phases), use extract_discovery_metadata()
    separately as needed.

    Returns:
        Expression resolving to (epjson_path, actuator_descriptions)
    """
    epjson = prepare_building(input_file, energyplus_path, src_version)
    return make_controllable(epjson)


__all__ = [
    # Entry points
    "create_complete_pipeline",
    "prepare_building",
    "make_controllable",
    # Types
    "ActuatorDescription",
    "Metadata",
    # Discovery metadata
    "extract_discovery_metadata",
    "Metadata",
    # Conversion / upgrade
    "Transition",
    "all_transitions",
    "scan_upgraders",
    "upgrade_idf",
    "upgrade",
    "ConvertIDF",
    "convert_idf",
    # Output and simulation steps
    "add_all_outputs",
    "add_hvac_meters",
    "add_outdoor_air_meters",
    "add_edd_output",
    "add_sqlite_output",
    "add_tabular_output",
    "modify_timestep",
    "detect_warmup_phases",
    "run_simulation",
    "eplustbl",
    "eddfile",
    "eiofile",
    "link_in_schedule",
    # Generic building edits
    "AddSetpointControl",
    "add_setpoint_control",
    "get_temperature_setpoints",
    "GlueSurfaces",
    "glue_surfaces",
    # Parsers / utilities
    "get_net_conditioned_area",
    "get_warmup_days",
    "get_hvac_actuators",
    "get_schedule_value_actuators",
    "get_b2b_scheduled_node_setpoint_actuators",
    "get_zone_temperature_control_actuators",
    "EddActuatorDescriptor",
    "iter_edd_actuators",
    "get_airflow_and_coil_node_setpoint_actuators",
]
