from __future__ import annotations

from b2b.baselines.unitary_actuators import select_unitary_actuator_indices


def test_select_unitary_actuator_indices_accepts_new_unitary_node_schedule_names() -> None:
    action_names = [
        "Fan::Fan Air Mass Flow Rate::mini split heat pump supply fan",
        "Schedule:Constant::Schedule Value::B2B unitaryhvac schedule for node (3)",
        "Schedule:Constant::Schedule Value::B2B unitaryhvac schedule for node (4)",
        "Schedule:Constant::Schedule Value::B2B unitaryhvac schedule for node (5)",
    ]
    idxs = select_unitary_actuator_indices(action_names)

    assert idxs.idx_fans == [0]
    # We don't know which node is which from the schedule name anymore, but we
    # should still pick *some* node-setpoint actuators so the baseline can run.
    assert sorted(idxs.idx_outlet_nodes) == [1, 2, 3]

