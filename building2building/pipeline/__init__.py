"""
Pipeline package.
"""

from pathlib import Path

from building2building.pipeline.actuators import ActuatorDescription, make_controllable
from building2building.pipeline.discovery import Metadata, extract_discovery_metadata
from building2building.pipeline.parse_edd import (
    EddActuatorDescriptor,
    get_airflow_and_coil_node_setpoint_actuators,
    get_b2b_scheduled_node_setpoint_actuators,
    get_hvac_actuators,
    get_schedule_value_actuators,
    get_zone_temperature_control_actuators,
    iter_edd_actuators,
)
from building2building.pipeline.parse_reports import (
    get_net_conditioned_area,
    get_warmup_days,
)
from building2building.pipeline.simulation import (
    detect_warmup_phases,
    eddfile,
    eiofile,
    eplustbl,
    run_simulation,
)
from building2building.pipeline.steps.conversion import (
    ConvertIDF,
    Transition,
    all_transitions,
    convert_idf,
    scan_upgraders,
    upgrade,
    upgrade_idf,
)
from building2building.pipeline.steps.outputs import (
    add_all_outputs,
    add_edd_output,
    add_hvac_meters,
    add_outdoor_air_meters,
    add_sqlite_output,
    add_tabular_output,
    modify_timestep,
    modify_run_period,
)
from building2building.pipeline.steps.schedule_files import link_in_schedule
from building2building.pipeline.steps.surfaces import GlueSurfaces, glue_surfaces
from building2building.store import Derivation, Expression, Realizable, Rename


def prepare_building(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
    timesteps_per_hour: int = 12,
) -> Derivation:
    """
    Convert IDF to epJSON and prepare for controllability.

    Steps:
    1. Upgrade IDF to target EnergyPlus version
    2. Convert IDF → epJSON
    3. Add HVAC meters (electricity, gas) - needed for RL reward calculation
    4. Add outdoor air meters - needed for RL observations
    5. Modify timestep to *timesteps_per_hour* steps/hour

    Does NOT add discovery outputs (EDD, tabular, SQLite) or modify HVAC control.
    Use make_controllable() and extract_discovery_metadata() for those.

    Returns:
        Derivation resolving to prepared epJSON
    """
    current = upgrade(input_file, energyplus_path, src_version)
    current = convert_idf(current, energyplus_path)
    current = add_hvac_meters(current)
    current = add_outdoor_air_meters(current)
    current = modify_timestep(current, timesteps_per_hour=timesteps_per_hour)
    current = modify_run_period(
        current, begin_day_of_month=1, begin_month=1, end_day_of_month=31, end_month=12
    )
    current = Rename("building.epjson", current)
    return current


def create_complete_pipeline(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
    timesteps_per_hour: int = 12,
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
    epjson = prepare_building(
        input_file,
        energyplus_path,
        src_version,
        timesteps_per_hour=timesteps_per_hour,
    )
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
