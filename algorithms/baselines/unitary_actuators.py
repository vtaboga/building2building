from __future__ import annotations

from dataclasses import dataclass

from algorithms.baselines.common import find_action_indices


@dataclass(frozen=True, slots=True)
class UnitaryActuatorIndices:
    idx_avail: list[int]
    idx_fans: list[int]
    idx_heat_nodes: list[int]
    idx_supp_nodes: list[int]
    idx_cool_nodes: list[int]
    idx_outlet_nodes: list[int]


def select_unitary_actuator_indices(action_names: list[str]) -> UnitaryActuatorIndices:
    """
    Select actuator indices for unitary HVAC control mode:
    - availability override
    - fan air mass flow
    - outlet (or "outlet-like") node temperature setpoints

    We support two equivalent encodings for outlet setpoints:
    - Preferred: `Schedule:Constant` / `Schedule Value` actuators created by
      `make_unitary_hvac_controllable` (B2B scheduled node setpoints). Note:
      recent versions name these schedules as "B2B unitaryhvac schedule for node (N)"
      (no longer encoding the node role in the schedule name).
    - Fallback: System Node Setpoint / Temperature Setpoint actuators (fixtures / legacy)
    """
    idx_avail = find_action_indices(
        action_names,
        component_type_prefix="airloophvac",
        control_type="availability status",
    )
    idx_fans = find_action_indices(
        action_names,
        component_type_prefix="fan",
        control_type="fan air mass flow rate",
    )

    # 1) Preferred: scheduled setpoints (Schedule Value actuators)
    #
    # Legacy naming (role encoded in the schedule name).
    idx_heat_nodes_sched = find_action_indices(
        action_names,
        component_type_prefix="schedule:",
        control_type="schedule value",
        component_name_contains="b2b node temp sp heating_coil",
    )
    idx_supp_nodes_sched = find_action_indices(
        action_names,
        component_type_prefix="schedule:",
        control_type="schedule value",
        component_name_contains="b2b node temp sp supplemental_coil",
    )
    idx_cool_nodes_sched = find_action_indices(
        action_names,
        component_type_prefix="schedule:",
        control_type="schedule value",
        component_name_contains="b2b node temp sp cooling_coil",
    )
    idx_outlet_nodes_sched = find_action_indices(
        action_names,
        component_type_prefix="schedule:",
        control_type="schedule value",
        component_name_contains="b2b node temp sp unitary_outlet",
    )

    # Current naming: schedules are created per node, but without embedding the
    # node role in the schedule name. In that case, we conservatively treat all
    # unitaryhvac node schedules as "outlet-like" setpoints so the baseline can
    # still hold all relevant node setpoints at a reasonable fixed value.
    idx_unitary_nodes_sched = find_action_indices(
        action_names,
        component_type_prefix="schedule:constant",
        control_type="schedule value",
        component_name_contains="b2b unitaryhvac schedule for node",
    )
    if (
        not idx_heat_nodes_sched
        and not idx_supp_nodes_sched
        and not idx_cool_nodes_sched
        and not idx_outlet_nodes_sched
        and idx_unitary_nodes_sched
    ):
        idx_outlet_nodes_sched = idx_unitary_nodes_sched

    # 2) Fallback: direct system node setpoint actuators (fixtures / legacy)
    idx_heat_nodes_node = find_action_indices(
        action_names,
        component_type_prefix="system node setpoint",
        control_type="temperature setpoint",
        component_name_contains="heating coil node",
    )
    idx_supp_nodes_node = find_action_indices(
        action_names,
        component_type_prefix="system node setpoint",
        control_type="temperature setpoint",
        component_name_contains="supplemental coil node",
    )
    idx_cool_nodes_node = find_action_indices(
        action_names,
        component_type_prefix="system node setpoint",
        control_type="temperature setpoint",
        component_name_contains="cooling coil node",
    )

    idx_outlet_nodes_node: list[int] = []
    idx_all_node_setpoints = find_action_indices(
        action_names,
        component_type_prefix="system node setpoint",
        control_type="temperature setpoint",
    )
    for i in idx_all_node_setpoints:
        name = str(action_names[int(i)]).lower()
        if "node" in name and "coil node" not in name:
            idx_outlet_nodes_node.append(int(i))

    # Choose scheduled if present, otherwise fallback to node actuators.
    use_scheduled = bool(
        idx_heat_nodes_sched
        or idx_supp_nodes_sched
        or idx_cool_nodes_sched
        or idx_outlet_nodes_sched
    )
    if use_scheduled:
        idx_heat_nodes = idx_heat_nodes_sched
        idx_supp_nodes = idx_supp_nodes_sched
        idx_cool_nodes = idx_cool_nodes_sched
        idx_outlet_nodes = idx_outlet_nodes_sched
    else:
        idx_heat_nodes = idx_heat_nodes_node
        idx_supp_nodes = idx_supp_nodes_node
        idx_cool_nodes = idx_cool_nodes_node
        idx_outlet_nodes = idx_outlet_nodes_node

    if not idx_fans or (
        not idx_outlet_nodes
    ):
        raise RuntimeError(
            "Baseline unitary PI controller could not find required actuators in action_names. "
            "Expected at least one Fan::Fan Air Mass Flow Rate and at least one "
            "outlet (or outlet-like) node temperature setpoint actuator (scheduled or direct).\n"
            f"action_names={action_names}"
        )

    return UnitaryActuatorIndices(
        idx_avail=idx_avail,
        idx_fans=idx_fans,
        idx_heat_nodes=idx_heat_nodes,
        idx_supp_nodes=idx_supp_nodes,
        idx_cool_nodes=idx_cool_nodes,
        idx_outlet_nodes=idx_outlet_nodes,
    )

