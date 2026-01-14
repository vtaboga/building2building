"""
Pipeline package.
"""

from building2building.pipeline.entrypoints import create_complete_pipeline
from building2building.pipeline.pipelines import (
    create_control_pipeline,
    create_discovery_pipeline,
)
from building2building.pipeline.simulation import eddfile, eiofile, eplustbl, run_simulation
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
    AddEDDOutput,
    AddHVACMeters,
    AddOutdoorAirMeters,
    AddTabularOutput,
    ModifyTimestep,
    add_edd_output,
    add_hvac_meters,
    add_outdoor_air_meters,
    add_tabular_output,
    modify_timestep,
)
from building2building.pipeline.steps.schedule_files import link_in_schedule
from building2building.pipeline.steps.surfaces import GlueSurfaces, glue_surfaces
from building2building.pipeline.steps.thermostat_setpoints import (
    AddSetpointControl,
    add_setpoint_control,
    get_temperature_setpoints,
)
from building2building.pipeline.hvac.baseboard import (
    AddBaseboardAvailabilityControl,
    get_baseboard_availability_schedule_names,
)
from building2building.pipeline.hvac.unitary import (
    AddNodeSetpointDiagnostics,
    EnsureScheduledNodeTemperatureSetpoints,
    SetUnitarySystemsToSetpointControl,
    add_node_setpoint_diagnostics,
    add_node_setpoint_diagnostics_inplace,
    ensure_scheduled_node_temperature_setpoints,
    ensure_scheduled_node_temperature_setpoints_inplace,
    get_b2b_scheduled_setpoint_schedule_names,
    get_unitary_air_outlet_node_names,
    set_unitary_systems_to_setpoint_control,
    set_unitary_systems_to_setpoint_control_inplace,
)
from building2building.pipeline.parse_reports import get_net_conditioned_area, get_warmup_days
from building2building.pipeline.parse_edd import (
    EddActuatorDescriptor,
    get_airflow_and_coil_node_setpoint_actuators,
    get_b2b_scheduled_node_setpoint_actuators,
    get_hvac_actuators,
    get_schedule_value_actuators,
    get_zone_temperature_control_actuators,
    iter_edd_actuators,
)

__all__ = [
    # Entry points
    "create_complete_pipeline",
    "create_discovery_pipeline",
    "create_control_pipeline",
    # Conversion / upgrade
    "Transition",
    "all_transitions",
    "scan_upgraders",
    "upgrade_idf",
    "upgrade",
    "ConvertIDF",
    "convert_idf",
    # Output and simulation steps
    "AddHVACMeters",
    "add_hvac_meters",
    "AddOutdoorAirMeters",
    "add_outdoor_air_meters",
    "AddEDDOutput",
    "add_edd_output",
    "AddTabularOutput",
    "add_tabular_output",
    "ModifyTimestep",
    "modify_timestep",
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
    # HVAC system helpers (unitary + baseboard)
    "SetUnitarySystemsToSetpointControl",
    "set_unitary_systems_to_setpoint_control",
    "set_unitary_systems_to_setpoint_control_inplace",
    "EnsureScheduledNodeTemperatureSetpoints",
    "ensure_scheduled_node_temperature_setpoints",
    "ensure_scheduled_node_temperature_setpoints_inplace",
    "AddNodeSetpointDiagnostics",
    "add_node_setpoint_diagnostics",
    "add_node_setpoint_diagnostics_inplace",
    "get_unitary_air_outlet_node_names",
    "get_b2b_scheduled_setpoint_schedule_names",
    "AddBaseboardAvailabilityControl",
    "get_baseboard_availability_schedule_names",
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

