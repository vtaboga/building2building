from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


class Policy(ABC):
    @abstractmethod
    def predict(self, observation, deterministic: bool = True):
        raise NotImplementedError


class ConstantPolicy(Policy):
    """Dummy policy that always returns the same action."""

    def __init__(self, actions=None):
        self.actions = actions

    def set_actions(self, actions):
        print(f"setting actions: {actions}")
        self.actions = actions

    def predict(self, observation, deterministic: bool = None):
        return self.actions, None


class ConstantSetPoints(Policy):
    def __init__(self, heating_setpoint: float, delta_setpoint: float):
        self.heating_setpoint = heating_setpoint
        self.delta_setpoint = delta_setpoint

    def predict(self, observation, deterministic: bool = True):
        action = [self.heating_setpoint, self.delta_setpoint]
        return action, None


@dataclass(frozen=True)
class OnOffSensibleLoadPolicy(Policy):
    """
    Simple thermostat-like on/off controller for a 1D action space:
    "Unitary HVAC :: Sensible Load Request" [W].

    Positive = heating request, negative = cooling request, 0 = off.
    """

    target_temp_c: float
    deadband_c: float
    q_heat_w: float
    q_cool_w: float
    temp_obs_index: int
    last_mode: str = "off"

    def predict(self, observation: Any, deterministic: bool = True):
        obs = np.asarray(observation, dtype=float).reshape(-1)
        tz = float(obs[self.temp_obs_index])

        if tz < self.target_temp_c - self.deadband_c:
            object.__setattr__(self, "last_mode", "heat")
            return np.asarray([float(self.q_heat_w)], dtype=float), None
        if tz > self.target_temp_c + self.deadband_c:
            object.__setattr__(self, "last_mode", "cool")
            return np.asarray([-float(self.q_cool_w)], dtype=float), None

        object.__setattr__(self, "last_mode", "off")
        return np.asarray([0.0], dtype=float), None


@dataclass(frozen=True)
class HVACActuatorOnOffPolicy(Policy):
    """
    On/off control using HVAC component actuators (fan/coil/airloop availability).

    This policy is designed for the action space created by
    `building2building.simulator.action_spaces.hvac_actuators_transform`, where
    `env.metadata['action_names']` aligns 1:1 with the action vector indices.
    """

    target_temp_c: float
    deadband_c: float
    # Actuator command levels (heuristic defaults; tweak per-building)
    fan_mass_flow_kg_s: float = 1.0
    terminal_mass_flow_kg_s: float = 1.0
    coil_speed_heat: float = 1.0
    coil_speed_cool: float = 1.0
    supplemental_stage_heat: float = 0.0
    # Optional: if the model exposes a Unitary HVAC "Sensible Load Request" actuator,
    # we can force heating/cooling demand directly (bypassing thermostat setpoints).
    q_heat_w: float = 0.0
    q_cool_w: float = 0.0
    availability_on: float = 2.0  # CycleOn
    availability_off: float = 1.0  # ForceOff
    last_mode: str = "off"

    # Filled by `from_env_metadata`
    n_actions: int = 0
    zone_temp_indices: tuple[int, ...] = ()
    idx_airloop_availability: Optional[int] = None
    idx_fan_mass_flow: Optional[int] = None
    idx_terminal_mass_flow: Optional[int] = None
    idx_coil_speed_value: Optional[int] = None
    idx_supplemental_stage: Optional[int] = None
    idx_unitary_sensible_load_request: Optional[int] = None

    @staticmethod
    def _find_zone_temp_indices(observation_names: Sequence[str]) -> tuple[int, ...]:
        idxs: list[int] = []
        for i, name in enumerate(observation_names):
            if "zone air temperature" in str(name).lower():
                idxs.append(i)
        if not idxs:
            raise ValueError("Could not find any 'Zone Air Temperature' entries in observation_names")
        return tuple(idxs)

    @staticmethod
    def _find_actuator_index(action_names: Sequence[str], predicate) -> Optional[int]:
        for i, nm in enumerate(action_names):
            if predicate(str(nm)):
                return i
        return None

    @classmethod
    def from_env_metadata(
        cls,
        *,
        env_metadata: dict[str, Any],
        target_temp_c: float,
        deadband_c: float,
        fan_mass_flow_kg_s: float = 1.0,
        terminal_mass_flow_kg_s: float = 1.0,
        coil_speed_heat: float = 1.0,
        coil_speed_cool: float = 1.0,
        supplemental_stage_heat: float = 0.0,
        q_heat_w: float = 0.0,
        q_cool_w: float = 0.0,
        availability_on: float = 2.0,
        availability_off: float = 1.0,
    ) -> "HVACActuatorOnOffPolicy":
        obs_names_raw = env_metadata.get("observation_names")
        act_names_raw = env_metadata.get("action_names")

        if not isinstance(obs_names_raw, list) or not all(isinstance(x, str) for x in obs_names_raw):
            raise TypeError("env.metadata['observation_names'] must be a list[str]")
        if not isinstance(act_names_raw, list) or not all(isinstance(x, str) for x in act_names_raw):
            raise TypeError("env.metadata['action_names'] must be a list[str]")

        zone_idxs = cls._find_zone_temp_indices(obs_names_raw)
        n_actions = len(act_names_raw)

        def has_component(component: str, control: str):
            c = component.lower()
            k = control.lower()

            def _pred(nm: str) -> bool:
                parts = nm.split("::")
                if len(parts) < 2:
                    return False
                return parts[0].strip().lower() == c and parts[1].strip().lower() == k

            return _pred

        idx_airloop = cls._find_actuator_index(
            act_names_raw, has_component("AirLoopHVAC", "Availability Status")
        )
        idx_fan = cls._find_actuator_index(
            act_names_raw, has_component("Fan", "Fan Air Mass Flow Rate")
        )
        idx_terminal = cls._find_actuator_index(
            act_names_raw,
            has_component("AirTerminal:SingleDuct:ConstantVolume:NoReheat", "Mass Flow Rate"),
        )
        idx_speed = cls._find_actuator_index(
            act_names_raw, has_component("Coil Speed Control", "Unitary System DX Coil Speed Value")
        )
        idx_supp = cls._find_actuator_index(
            act_names_raw,
            has_component("Coil Speed Control", "Unitary System Supplemental Coil Stage Level"),
        )
        idx_load = cls._find_actuator_index(
            act_names_raw, has_component("Unitary HVAC", "Sensible Load Request")
        )

        return cls(
            target_temp_c=target_temp_c,
            deadband_c=deadband_c,
            fan_mass_flow_kg_s=fan_mass_flow_kg_s,
            terminal_mass_flow_kg_s=terminal_mass_flow_kg_s,
            coil_speed_heat=coil_speed_heat,
            coil_speed_cool=coil_speed_cool,
            supplemental_stage_heat=supplemental_stage_heat,
            q_heat_w=q_heat_w,
            q_cool_w=q_cool_w,
            availability_on=availability_on,
            availability_off=availability_off,
            n_actions=n_actions,
            zone_temp_indices=zone_idxs,
            idx_airloop_availability=idx_airloop,
            idx_fan_mass_flow=idx_fan,
            idx_terminal_mass_flow=idx_terminal,
            idx_coil_speed_value=idx_speed,
            idx_supplemental_stage=idx_supp,
            idx_unitary_sensible_load_request=idx_load,
        )

    def _mean_zone_temp(self, observation: Any) -> float:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        temps = [float(obs[i]) for i in self.zone_temp_indices]
        return float(np.mean(np.asarray(temps, dtype=float)))

    def predict(self, observation: Any, deterministic: bool = True):
        tz = self._mean_zone_temp(observation)
        if self.n_actions <= 0:
            raise RuntimeError(
                "Policy is not configured with n_actions (action_names missing?). "
                "Use HVACActuatorOnOffPolicy.from_env_metadata()."
            )

        act = np.zeros((int(self.n_actions),), dtype=float)

        def set_if(idx: Optional[int], value: float) -> None:
            if idx is None:
                return
            act[idx] = float(value)

        if tz < self.target_temp_c - self.deadband_c:
            object.__setattr__(self, "last_mode", "heat")
            set_if(self.idx_airloop_availability, self.availability_on)
            set_if(self.idx_fan_mass_flow, self.fan_mass_flow_kg_s)
            set_if(self.idx_terminal_mass_flow, self.terminal_mass_flow_kg_s)
            set_if(self.idx_coil_speed_value, self.coil_speed_heat)
            set_if(self.idx_supplemental_stage, self.supplemental_stage_heat)
            if self.q_heat_w > 0.0:
                set_if(self.idx_unitary_sensible_load_request, float(self.q_heat_w))
        elif tz > self.target_temp_c + self.deadband_c:
            object.__setattr__(self, "last_mode", "cool")
            set_if(self.idx_airloop_availability, self.availability_on)
            set_if(self.idx_fan_mass_flow, self.fan_mass_flow_kg_s)
            set_if(self.idx_terminal_mass_flow, self.terminal_mass_flow_kg_s)
            set_if(self.idx_coil_speed_value, self.coil_speed_cool)
            set_if(self.idx_supplemental_stage, 0.0)
            if self.q_cool_w > 0.0:
                set_if(self.idx_unitary_sensible_load_request, -float(self.q_cool_w))
        else:
            object.__setattr__(self, "last_mode", "off")
            set_if(self.idx_airloop_availability, self.availability_off)
            set_if(self.idx_fan_mass_flow, 0.0)
            set_if(self.idx_terminal_mass_flow, 0.0)
            set_if(self.idx_coil_speed_value, 0.0)
            set_if(self.idx_supplemental_stage, 0.0)
            # If we are overriding load request, explicitly clear it to 0W.
            set_if(self.idx_unitary_sensible_load_request, 0.0)

        return act, None