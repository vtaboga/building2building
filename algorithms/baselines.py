from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def predict(self, observation, deterministic: bool = True):
        raise NotImplementedError


class ConstantPolicy(Policy):
    def __init__(self, heating_setpoint: float, delta_setpoint: float):
        self.heating_setpoint = heating_setpoint
        self.delta_setpoint = delta_setpoint

    def predict(self, observation, deterministic: bool = True):
        action = [self.heating_setpoint, self.delta_setpoint]
        return action, None