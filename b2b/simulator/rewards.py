from dataclasses import dataclass

import numpy as np

from b2b.simulator.action_spaces import ThermostatSetpoint


def base_reward_function(
    obs,
    area: float,
    controlled_zones: list[str],
    energy_weight=1.0,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Args:
        obs: Dictionary containing observations
        setpoints: Dictionary mapping zones to their setpoint configurations
        building_characteristics: Dictionary containing building characteristics
        energy_weight: Weight for the energy consumption penalty
    Returns:
        float: Combined reward (negative values represent penalties)
    """

    # Energy consumption penalty (in Wh/floor area)
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]
    energy_penalty = energy_penalty / 3600.0 / area

    # Calculate temperature tracking error for controlled zones
    temp_error = 0
    target_temp = 21.0  # Target temperature in °C

    for zone in controlled_zones:
        current_temp = obs["temperature"][zone]
        temp_error += (current_temp - target_temp) ** 2

    temp_error = temp_error / max(len(controlled_zones), 1)

    # Combine rewards (negative values represent penalties)
    # Equal weighting between temperature tracking and energy consumption
    total_reward = -(temp_error + energy_weight * energy_penalty)

    return total_reward


@dataclass
class BaseReward:
    area: float
    controlled_zones: list[str]
    energy_weight: float

    def __call__(self, obs):
        return base_reward_function(
            obs, self.area, self.controlled_zones, self.energy_weight
        )


def barrier_reward_function(
    obs,
    area: float,
    controlled_zones: list[str],
    energy_weight=1.0,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Args:
        obs: Dictionary containing observations
        setpoints: Dictionary mapping zones to their setpoint configurations
        building_characteristics: Dictionary containing building characteristics

    Returns:
        float: Combined reward (negative values represent penalties)
    """

    # Energy consumption penalty (in Wh/floor area)
    energy_penalty = (
        obs["energy"]["HVAC_electricity"] + obs["energy"]["HVAC_natural_gas"]
    )
    energy_penalty = energy_penalty / 3600.0 / area

    # Calculate temperature tracking error for controlled zones
    temp_error = 0
    target_temp = 21.0  # Target temperature in °C
    delta_temp = 1.5  # Deadband in °C
    penalty = 1000  # Penalty for temperature error outside of deadband

    for zone in controlled_zones:
        current_temp = obs["temperature"][zone]
        if abs(current_temp - target_temp) > delta_temp:
            temp_error += penalty

    # Combine rewards (negative values represent penalties)
    total_reward = -(temp_error + energy_weight * energy_penalty)

    return total_reward


@dataclass
class BarrierReward:
    area: float
    controlled_zones: list[str]
    energy_weight: float

    def __call__(self, obs):
        return barrier_reward_function(
            obs, self.area, self.controlled_zones, self.energy_weight
        )


def deadband_reward_function(
    obs,
    area: float,
    controlled_zones: list[str],
    energy_weight=1.0,
    target_temp: float = 21.0,
    dT: float = 0.5,
) -> float:
    # Energy consumption penalty (in Wh/floor area)
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]
    energy_penalty = energy_penalty / 3600.0 / area

    # Comfort: temperature error for controlled zones
    temp_error = 0

    for zone in controlled_zones:
        current_temp = obs["temperature"][zone]
        temp_error += np.max([0, np.abs(current_temp - target_temp) - dT])

    temp_error = temp_error / len(controlled_zones)

    comfort_penalty = temp_error

    total_reward = -(comfort_penalty + energy_weight * energy_penalty)

    return total_reward


@dataclass
class DeadbandReward:
    area: float
    controlled_zones: list[str]
    energy_weight: float
    target_temp: float
    dT: float

    def __call__(self, obs):
        return deadband_reward_function(
            obs=obs,
            area=self.area,
            controlled_zones=self.controlled_zones,
            energy_weight=self.energy_weight,
            target_temp=self.target_temp,
            dT=self.dT,
        )
