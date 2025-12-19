from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


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