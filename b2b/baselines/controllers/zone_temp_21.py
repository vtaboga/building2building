"""Rule-based controller that maintains all zones around 21°C.

Uses Zone Temperature Control (heating/cooling setpoints), VAV Primary Air
Maximum Flow Fraction, or System Node Setpoint (Temperature/Mass Flow Rate)
actuators. Suitable for multi-zone buildings with AirTerminal:SingleDuct:VAV:Reheat.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from b2b.baselines.common import find_action_indices, require_env_metadata_list_str


def _fill_defaults(action: np.ndarray, default: float = 0.0) -> None:
    """Fill action array with default where no specific command is set."""
    action[:] = default


class ZoneTemp21Policy:
    """
    Simple rule-based policy: set all zone thermostats to 21°C (±0.5°C deadband)
    and keep VAV dampers fully open.

    Exposes an SB3-like `predict()` API for use with benchmark rollout executors.
    """

    def __init__(self, policy_cfg: Any) -> None:
        self.target_temp_c = float(getattr(policy_cfg, "target_temp_c", 21.0))
        self.deadband_c = float(getattr(policy_cfg, "deadband_c", 0.5))
        self.heat_sp = float(
            getattr(policy_cfg, "heating_setpoint_c", self.target_temp_c - self.deadband_c)
        )
        self.cool_sp = float(
            getattr(policy_cfg, "cooling_setpoint_c", self.target_temp_c + self.deadband_c)
        )
        self.vav_fraction = float(getattr(policy_cfg, "vav_max_flow_fraction", 1.0))
        self.vav_mass_flow_kg_s = float(
            getattr(policy_cfg, "vav_default_mass_flow_kg_s", 1.0)
        )
        self._act_names: list[str] = []
        self._n_act = 0
        self._idx_heat: list[int] = []
        self._idx_cool: list[int] = []
        self._idx_vav: list[int] = []
        self._idx_node_temp: list[int] = []
        self._idx_node_massflow: list[int] = []
        self._idx_other: list[int] = []

    def bind_env(self, env: Any) -> None:
        act_names = require_env_metadata_list_str(env, "action_names")
        self._act_names = act_names
        self._n_act = len(act_names)

        self._idx_heat = find_action_indices(
            act_names,
            component_type_prefix="zone temperature control",
            control_type="heating setpoint",
        )
        self._idx_cool = find_action_indices(
            act_names,
            component_type_prefix="zone temperature control",
            control_type="cooling setpoint",
        )
        self._idx_vav = find_action_indices(
            act_names,
            component_type_prefix="airterminal:singleduct:vav:reheat",
            control_type="primary air maximum flow fraction",
        )
        self._idx_node_temp = find_action_indices(
            act_names,
            component_type_prefix="system node setpoint",
            control_type="temperature setpoint",
        )
        self._idx_node_massflow = find_action_indices(
            act_names,
            component_type_prefix="system node setpoint",
            control_type="mass flow rate setpoint",
        )

        assigned: set[int] = set(
            self._idx_heat
            + self._idx_cool
            + self._idx_vav
            + self._idx_node_temp
            + self._idx_node_massflow
        )
        self._idx_other = [i for i in range(self._n_act) if i not in assigned]

        if not (
            self._idx_heat
            or self._idx_cool
            or self._idx_vav
            or self._idx_node_temp
            or self._idx_node_massflow
        ):
            raise RuntimeError(
                "ZoneTemp21Policy requires Zone Temperature Control, VAV Reheat, "
                "or System Node Setpoint actuators. "
                f"Found none in action_names={act_names[:20]}..."
            )

    def predict(
        self, obs: Any, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        action = np.zeros((self._n_act,), dtype=float)
        _fill_defaults(action)

        for i in self._idx_heat:
            action[int(i)] = float(self.heat_sp)
        for i in self._idx_cool:
            action[int(i)] = float(self.cool_sp)
        for i in self._idx_vav:
            action[int(i)] = float(self.vav_fraction)
        for i in self._idx_node_temp:
            action[int(i)] = float(self.target_temp_c)
        for i in self._idx_node_massflow:
            action[int(i)] = float(self.vav_mass_flow_kg_s)

        for i in self._idx_other:
            name = str(self._act_names[int(i)]).lower()
            if "availability" in name:
                action[int(i)] = 1.0
            elif "schedule value" in name and "temperature" in name:
                action[int(i)] = float(self.target_temp_c)

        return action, None
