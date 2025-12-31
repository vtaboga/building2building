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


@dataclass(slots=True)
class PIDSensibleLoadPolicy(Policy):
    """
    PID controller for a 1D action space: "Unitary HVAC :: Sensible Load Request" [W].

    Sign convention:
    - Positive output requests heating
    - Negative output requests cooling

    Notes:
    - This policy assumes the observation contains a zone air temperature at `temp_obs_index`.
    - `dt_s` is the control interval in seconds. If you're stepping at fixed timestep,
      leave it constant. If unknown, leaving it at 1.0 still yields a stable controller
      after retuning gains.
    """

    target_temp_c: float
    deadband_c: float
    temp_obs_index: int

    # PID gains on error e = target - tz
    kp: float
    ki: float
    kd: float

    # Output saturation (Watts)
    q_heat_max_w: float
    q_cool_max_w: float

    # Control interval
    dt_s: float = 1.0

    # Anti-windup: clamp integral of error (in C*s)
    integral_min: float = -1e6
    integral_max: float = 1e6

    # State
    integral: float = 0.0
    prev_error: float | None = None
    last_mode: str = "off"

    def predict(self, observation: Any, deterministic: bool = True):
        obs = np.asarray(observation, dtype=float).reshape(-1)
        tz = float(obs[self.temp_obs_index])
        error = float(self.target_temp_c - tz)

        # Deadband: explicitly off. Also bleed integral slowly to avoid latch-up.
        if abs(error) <= float(self.deadband_c):
            self.integral *= 0.9
            object.__setattr__(self, "last_mode", "off")
            object.__setattr__(self, "prev_error", error)
            return np.asarray([0.0], dtype=float), None

        dt = float(self.dt_s)
        if dt <= 0:
            raise ValueError("dt_s must be > 0")

        # Integrate error
        self.integral = float(self.integral + error * dt)
        self.integral = float(np.clip(self.integral, self.integral_min, self.integral_max))

        # Derivative
        if self.prev_error is None:
            d_error = 0.0
        else:
            d_error = float((error - self.prev_error) / dt)

        u = float(self.kp * error + self.ki * self.integral + self.kd * d_error)

        # Saturate to physically meaningful ranges.
        u = float(np.clip(u, -abs(self.q_cool_max_w), abs(self.q_heat_max_w)))

        object.__setattr__(self, "prev_error", error)
        object.__setattr__(self, "last_mode", "heat" if u > 0 else "cool")
        return np.asarray([u], dtype=float), None


@dataclass(slots=True)
class TrimAndRespondSensibleLoadPolicy(Policy):
    """
    Trim-and-Respond controller for a sensible load request actuator.

    Control idea:
    - Maintain an internal "current load request" (W).
    - If temperature is outside deadband, "respond" by stepping load up/down.
    - If temperature is inside deadband, "trim" the load back toward zero.

    This is often easier to tune than PID for systems with long time constants.
    """

    target_temp_c: float
    deadband_c: float
    temp_obs_index: int

    # Step sizes (W) for response and trim
    respond_step_w: float
    trim_step_w: float

    # Output saturation (W)
    q_heat_max_w: float
    q_cool_max_w: float

    # State
    current_load_w: float = 0.0
    last_mode: str = "off"

    def predict(self, observation: Any, deterministic: bool = True):
        obs = np.asarray(observation, dtype=float).reshape(-1)
        tz = float(obs[self.temp_obs_index])
        err = float(self.target_temp_c - tz)

        if abs(err) <= float(self.deadband_c):
            # Trim toward 0
            if self.current_load_w > 0.0:
                self.current_load_w = max(0.0, self.current_load_w - float(self.trim_step_w))
            elif self.current_load_w < 0.0:
                self.current_load_w = min(0.0, self.current_load_w + float(self.trim_step_w))
            object.__setattr__(self, "last_mode", "off" if self.current_load_w == 0.0 else self.last_mode)
            return np.asarray([float(self.current_load_w)], dtype=float), None

        # Respond: step in the direction of the error
        if err > 0.0:
            self.current_load_w = float(self.current_load_w + float(self.respond_step_w))
            self.current_load_w = float(min(self.current_load_w, abs(self.q_heat_max_w)))
            object.__setattr__(self, "last_mode", "heat")
        else:
            self.current_load_w = float(self.current_load_w - float(self.respond_step_w))
            self.current_load_w = float(max(self.current_load_w, -abs(self.q_cool_max_w)))
            object.__setattr__(self, "last_mode", "cool")

        return np.asarray([float(self.current_load_w)], dtype=float), None