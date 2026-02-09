from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from b2b.baselines.common import (
    find_controlled_zone_air_temp_index,
    require_env_metadata_list_str,
)
from b2b.baselines.unitary_actuators import select_unitary_actuator_indices


def compute_fan_command_constant(*, fan_mass_flow_kg_s: float) -> float:
    return float(fan_mass_flow_kg_s)


@dataclass(frozen=True, slots=True)
class FanCoilConstantMetrics:
    target_temp_c: float
    conditioned_zone_temp_c: float


class FanCoilConstantPolicy:
    """
    Constant actuator baseline.

    Exposes an SB3-like `predict()` API for use with benchmark rollout executors.
    """

    def __init__(self, policy_cfg: Any):
        self.target_temp_c = float(getattr(policy_cfg, "target_temp_c", 21.0))
        self.availability_on = float(getattr(policy_cfg, "availability_on", 2.0))

        self.fan_mass_flow_kg_s = float(getattr(policy_cfg, "fan_mass_flow_kg_s", 1.0))

        # Coil / node setpoints (if present in the action space).
        self.heating_coil_setpoint_c = float(
            getattr(policy_cfg, "heating_coil_setpoint_c", 45.0)
        )
        self.supplemental_coil_setpoint_c = float(
            getattr(policy_cfg, "supplemental_coil_setpoint_c", 45.0)
        )
        self.cooling_coil_setpoint_c = float(
            getattr(policy_cfg, "cooling_coil_setpoint_c", 18.0)
        )
        self.outlet_node_setpoint_c = float(
            getattr(policy_cfg, "outlet_node_setpoint_c", self.target_temp_c)
        )

        self._idxs = None
        self._temp_idx: int | None = None

    def bind_env(self, env: Any) -> None:
        obs_names = require_env_metadata_list_str(env, "observation_names")
        act_names = require_env_metadata_list_str(env, "action_names")

        controlled_zones = None
        if hasattr(env, "metadata") and isinstance(env.metadata, dict):
            cz = env.metadata.get("controlled_zones")
            if isinstance(cz, list) and all(isinstance(x, str) for x in cz):
                controlled_zones = cz

        self._idxs = select_unitary_actuator_indices(act_names)
        self._temp_idx = find_controlled_zone_air_temp_index(
            obs_names, controlled_zones=controlled_zones
        )

    def step_metrics(self, obs: Any, *, action: np.ndarray) -> dict[str, float]:
        if self._temp_idx is None:
            return {"target_temp_c": float(self.target_temp_c)}
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        tz = (
            float(obs_arr[int(self._temp_idx)])
            if int(self._temp_idx) < len(obs_arr)
            else float("nan")
        )
        return {
            "target_temp_c": float(self.target_temp_c),
            "conditioned_zone_temp_c": float(tz),
        }

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        if self._idxs is None:
            raise RuntimeError("Policy is not bound to an env; call bind_env(env) first.")

        n_act = max(
            max(self._idxs.idx_outlet_nodes, default=-1),
            max(self._idxs.idx_fans, default=-1),
            max(self._idxs.idx_avail, default=-1),
            max(self._idxs.idx_heat_nodes, default=-1),
            max(self._idxs.idx_supp_nodes, default=-1),
            max(self._idxs.idx_cool_nodes, default=-1),
        ) + 1
        action_cmd = np.zeros((int(n_act),), dtype=float)

        for idx in self._idxs.idx_avail:
            action_cmd[int(idx)] = float(self.availability_on)

        fan_cmd = compute_fan_command_constant(
            fan_mass_flow_kg_s=float(self.fan_mass_flow_kg_s)
        )
        for idx in self._idxs.idx_fans:
            action_cmd[int(idx)] = float(fan_cmd)

        for idx in self._idxs.idx_heat_nodes:
            action_cmd[int(idx)] = float(self.heating_coil_setpoint_c)
        for idx in self._idxs.idx_supp_nodes:
            action_cmd[int(idx)] = float(self.supplemental_coil_setpoint_c)
        for idx in self._idxs.idx_cool_nodes:
            action_cmd[int(idx)] = float(self.cooling_coil_setpoint_c)
        for idx in self._idxs.idx_outlet_nodes:
            action_cmd[int(idx)] = float(self.outlet_node_setpoint_c)

        return action_cmd, None

