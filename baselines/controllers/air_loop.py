"""Baseline controller for VAV air-loop HVAC systems.

Designed for multi-zone buildings with central air handling (e.g. medium
office with 3 VAV loops x 5 zones each).  Auto-discovers loops, zones,
and actuator indices from ``env.metadata["hvac_equipment"]``.

Control strategy:
  SAT  -- continuous P-controller biased toward warmest/coldest zone.
  Flow -- directional PI with rate limiting per zone; SAT-aware sign flip.
  Reheat setpoint -- per-zone heating demand for electric reheat coils.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from baselines.utils.metadata import (
    find_obs_index_optional,
    find_zone_air_temp_index,
)

# ---------------------------------------------------------------------------
# Config (plain dataclass, no OmegaConf dependency)
# ---------------------------------------------------------------------------


@dataclass
class AirLoopConfig:
    """Tunable parameters for the VAV air-loop baseline."""

    target_temp: float = 21.0
    deadband: float = 1.0
    sat_aware_flow: bool = True

    sat_neutral: float = 20.5
    sat_kp: float = 0.8
    sat_min: float = 10.0
    sat_max: float = 60.0
    sat_rate_limit: float = 0.3
    outdoor_sat_gain: float = 0.0
    sat_cold_bias: float = 0.0
    sat_warm_bias: float = 0.6

    flow_base: float = 0.4
    flow_kp: float = 0.2
    flow_ki: float = 0.025
    flow_min: float = 0.1
    flow_max: float = 1.0
    flow_rate_limit: float = 0.04
    integral_max: float = 15.0
    integral_decay: float = 0.99

    reheat_sp_min: float = 10.0
    reheat_sp_max: float = 25.0
    reheat_sp_deadband: float = 0.3
    reheat_sp_kp: float = 3.0
    reheat_sp_rate_limit: float = 0.2

    clg_sp_default: float = 22.0
    error_ema_alpha: float = 0.3

    # Per-loop outdoor-air mass-flow command written by the reactive
    # baseline at every step.  Pinned to the DOE OfficeMedium autosized
    # minimum ventilation flow (~1.12 m³/s × 1.225 kg/m³ ≈ 1.37 kg/s).
    # This knob is
    # intentionally NOT in the Optuna search space -- the agent is the
    # one learning to modulate OA, the RBC just provides a constant
    # floor against which agent learning is measured.
    oa_mass_flow: float = 1.37


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _ZoneState:
    zone_name: str
    temp_obs_idx: int
    flow_act_idx: int
    htg_act_idx: int
    clg_act_idx: int | None

    integral: float = 0.0
    smooth_error: float = 0.0
    prev_flow: float = 0.4
    prev_reheat_sp: float = 10.0


@dataclass
class _LoopState:
    sat_act_idx: int
    # Index of the per-loop OA-mixer actuator in the env's flat
    # ``action_names`` list.  Required: every VAVSystem produced by
    # ``make_vav_system_controllable`` carries an OA actuator.  Stale
    # equipment.json files lacking the OA actuator will already fail to
    # load via ``cattrs.structure`` before reaching this point.
    oa_act_idx: int
    zones: list[_ZoneState] = field(default_factory=list)
    prev_sat: float = 20.5


def _match_actuator(
    act_names: list[str],
    component_type: str,
    control_type: str,
    component_name: str,
    *,
    required: bool = True,
) -> int | None:
    target = f"{component_type}::{control_type}::{component_name}"
    for i, name in enumerate(act_names):
        if name == target:
            return i
    if required:
        raise RuntimeError(f"Action not found: {target!r}")
    return None


def _require_metadata_list_str(env: Any, key: str) -> list[str]:
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("env.metadata missing")
    raw = env.metadata.get(key)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"env.metadata['{key}'] must be a list[str]")
    return raw


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class AirLoopPolicy:
    """VAV baseline for central air-loop buildings.

    Usage::

        cfg = AirLoopConfig()
        policy = AirLoopPolicy(cfg)
        policy.bind_env(env)
        obs, _ = env.reset()
        action, _ = policy.predict(obs)
    """

    def __init__(self, cfg: AirLoopConfig | None = None) -> None:
        self.config = cfg or AirLoopConfig()
        self._loops: list[_LoopState] = []
        self._n_act: int = 0
        self._outdoor_temp_idx: int | None = None
        self._initialized: bool = False

    def bind_env(self, env: Any) -> None:
        """Wire observation/action indices from ``env.metadata``."""
        obs_names = _require_metadata_list_str(env, "observation_names")
        act_names = _require_metadata_list_str(env, "action_names")

        equipment = env.metadata.get("hvac_equipment", [])
        vav_systems = [
            e for e in equipment if getattr(e, "equipment_type", None) == "vavsystem"
        ]
        if not vav_systems:
            raise RuntimeError(
                "AirLoopPolicy requires VAVSystem equipment. "
                f"Found: {[type(e).__name__ for e in equipment]}"
            )

        self._outdoor_temp_idx = find_obs_index_optional(
            obs_names, "outdoor_temperature"
        )

        self._loops = []
        cfg = self.config

        for vav in vav_systems:
            sat_idx = _match_actuator(
                act_names,
                vav.supply_temp_setpoint.component_type,
                vav.supply_temp_setpoint.control_type,
                vav.supply_temp_setpoint.component_name,
            )
            assert sat_idx is not None

            oa_idx = _match_actuator(
                act_names,
                vav.oa_mass_flow.component_type,
                vav.oa_mass_flow.control_type,
                vav.oa_mass_flow.component_name,
            )
            assert oa_idx is not None

            zones: list[_ZoneState] = []
            for term in vav.terminals:
                flow_idx = _match_actuator(
                    act_names,
                    term.flow_fraction.component_type,
                    term.flow_fraction.control_type,
                    term.flow_fraction.component_name,
                )
                assert flow_idx is not None
                htg_idx = _match_actuator(
                    act_names,
                    term.heating_setpoint.component_type,
                    term.heating_setpoint.control_type,
                    term.heating_setpoint.component_name,
                )
                assert htg_idx is not None
                clg_idx = _match_actuator(
                    act_names,
                    term.cooling_setpoint.component_type,
                    term.cooling_setpoint.control_type,
                    term.cooling_setpoint.component_name,
                    required=False,
                )
                temp_idx = find_zone_air_temp_index(obs_names, term.zone)

                zones.append(
                    _ZoneState(
                        zone_name=term.zone,
                        temp_obs_idx=temp_idx,
                        flow_act_idx=flow_idx,
                        htg_act_idx=htg_idx,
                        clg_act_idx=clg_idx,
                        prev_flow=cfg.flow_base,
                        prev_reheat_sp=cfg.reheat_sp_min,
                    )
                )

            self._loops.append(
                _LoopState(
                    sat_act_idx=sat_idx,
                    oa_act_idx=oa_idx,
                    zones=zones,
                    prev_sat=cfg.sat_neutral,
                )
            )

        self._n_act = len(act_names)
        self._initialized = False
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        for loop in self._loops:
            loop.prev_sat = cfg.sat_neutral
            for z in loop.zones:
                z.integral = 0.0
                z.smooth_error = 0.0
                z.prev_flow = cfg.flow_base
                z.prev_reheat_sp = cfg.reheat_sp_min
        self._initialized = False

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        if not self._loops:
            raise RuntimeError("Not bound; call bind_env() first.")

        cfg = self.config
        obs_arr = np.asarray(obs, dtype=float).ravel()
        action = np.zeros(self._n_act, dtype=np.float64)

        t_outdoor = float("nan")
        if self._outdoor_temp_idx is not None and self._outdoor_temp_idx < len(obs_arr):
            t_outdoor = float(obs_arr[self._outdoor_temp_idx])

        outdoor_offset = 0.0
        if not np.isnan(t_outdoor):
            outdoor_offset = cfg.outdoor_sat_gain * (cfg.target_temp - t_outdoor)

        for loop in self._loops:
            raw_errors = np.array(
                [
                    (
                        float(obs_arr[z.temp_obs_idx]) - cfg.target_temp
                        if z.temp_obs_idx < len(obs_arr)
                        else 0.0
                    )
                    for z in loop.zones
                ],
                dtype=np.float64,
            )

            alpha = cfg.error_ema_alpha
            for i, z in enumerate(loop.zones):
                if not self._initialized:
                    z.smooth_error = raw_errors[i]
                else:
                    z.smooth_error = (
                        alpha * raw_errors[i] + (1 - alpha) * z.smooth_error
                    )

            errors = np.array([z.smooth_error for z in loop.zones], dtype=np.float64)

            # SAT
            w_cold = cfg.sat_cold_bias
            w_warm = cfg.sat_warm_bias
            w_mean = 1.0 - w_cold - w_warm
            weighted_err = (
                w_cold * float(np.min(errors))
                + w_warm * float(np.max(errors))
                + w_mean * float(np.mean(errors))
            )
            sat_target = cfg.sat_neutral - cfg.sat_kp * weighted_err + outdoor_offset
            sat_target = np.clip(sat_target, cfg.sat_min, cfg.sat_max)
            delta = np.clip(
                sat_target - loop.prev_sat,
                -cfg.sat_rate_limit,
                cfg.sat_rate_limit,
            )
            sat = float(np.clip(loop.prev_sat + delta, cfg.sat_min, cfg.sat_max))
            action[loop.sat_act_idx] = sat
            loop.prev_sat = sat

            # OA mixer: pinned at the constant configured value.  Not
            # rate-limited or modulated -- the agent learns OA
            # modulation; the RBC supplies a constant minimum floor.
            action[loop.oa_act_idx] = cfg.oa_mass_flow

            # Flow
            for i, z in enumerate(loop.zones):
                t_zone = float(obs_arr[z.temp_obs_idx])
                in_cooling = cfg.sat_aware_flow and sat < t_zone

                z.integral *= cfg.integral_decay
                if in_cooling:
                    at_max = z.prev_flow >= cfg.flow_max - 0.01 and errors[i] > 0
                    at_min = z.prev_flow <= cfg.flow_min + 0.01 and errors[i] < 0
                else:
                    at_max = z.prev_flow >= cfg.flow_max - 0.01 and errors[i] < 0
                    at_min = z.prev_flow <= cfg.flow_min + 0.01 and errors[i] > 0
                if not at_max and not at_min:
                    z.integral += errors[i]
                z.integral = float(
                    np.clip(z.integral, -cfg.integral_max, cfg.integral_max)
                )

                flow_sign = -1.0 if in_cooling else 1.0
                flow_target = cfg.flow_base - flow_sign * (
                    cfg.flow_kp * errors[i] + cfg.flow_ki * z.integral
                )
                flow_target = float(np.clip(flow_target, cfg.flow_min, cfg.flow_max))
                d = float(
                    np.clip(
                        flow_target - z.prev_flow,
                        -cfg.flow_rate_limit,
                        cfg.flow_rate_limit,
                    )
                )
                flow = float(np.clip(z.prev_flow + d, cfg.flow_min, cfg.flow_max))
                action[z.flow_act_idx] = flow
                z.prev_flow = flow

            # Reheat
            for i, z in enumerate(loop.zones):
                err = errors[i]
                if err < -cfg.reheat_sp_deadband:
                    sp_target = cfg.target_temp + cfg.reheat_sp_kp * (
                        -err - cfg.reheat_sp_deadband
                    )
                    sp_target = min(sp_target, cfg.reheat_sp_max)
                else:
                    sp_target = cfg.reheat_sp_min

                sp_target = float(
                    np.clip(sp_target, cfg.reheat_sp_min, cfg.reheat_sp_max)
                )
                d = float(
                    np.clip(
                        sp_target - z.prev_reheat_sp,
                        -cfg.reheat_sp_rate_limit,
                        cfg.reheat_sp_rate_limit,
                    )
                )
                sp = float(
                    np.clip(
                        z.prev_reheat_sp + d,
                        cfg.reheat_sp_min,
                        cfg.reheat_sp_max,
                    )
                )
                action[z.htg_act_idx] = sp
                z.prev_reheat_sp = sp

                if z.clg_act_idx is not None:
                    action[z.clg_act_idx] = cfg.clg_sp_default

        self._initialized = True
        return action, None
