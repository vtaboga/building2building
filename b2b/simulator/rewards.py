from dataclasses import dataclass

import numpy as np


def base_reward_function(
    obs,
    controlled_zones: list[str],
    energy_weight=1.0,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Args:
        obs: Dictionary containing observations (energy values in Wh/m²)
        controlled_zones: List of controlled zone names
        energy_weight: Weight for the energy consumption penalty
    Returns:
        float: Combined reward (negative values represent penalties)
    """

    # Energy observations are already in Wh/m² (converted in observation_spaces.py)
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    # Calculate temperature tracking error for controlled zones
    temp_error = 0
    target_temp = 21.0  # Target temperature in °C

    for zone in controlled_zones:
        current_temp = obs["temperature"][zone]
        temp_error += (current_temp - target_temp) ** 2

    temp_error = temp_error / len(controlled_zones)

    # Combine rewards (negative values represent penalties)
    # Equal weighting between temperature tracking and energy consumption
    total_reward = -(temp_error + energy_weight * energy_penalty)

    return total_reward


@dataclass
class BaseReward:
    controlled_zones: list[str]
    energy_weight: float

    def __call__(self, obs):
        return base_reward_function(
            obs, self.controlled_zones, self.energy_weight
        )


def barrier_reward_function(
    obs,
    controlled_zones: list[str],
    energy_weight=1.0,
    deadband_c: float = 0.5,
    violation_penalty: float = 100.0,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Args:
        obs: Dictionary containing observations (energy values in Wh/m²)
        controlled_zones: List of controlled zone names
        energy_weight: Weight for the energy consumption penalty
        deadband_c: Comfort deadband in °C
        violation_penalty: Penalty for comfort violations

    Returns:
        float: Combined reward (negative values represent penalties)
    """

    # Energy observations are already in Wh/m² (converted in observation_spaces.py)
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    # Barrier on comfort around a (possibly zone-specific, occupancy-aware) target.
    has_violation = False
    for zone in controlled_zones:
        current_temp = float(obs["temperature"][zone])
        target_temp = 21.0
        if "target_temperature" in obs and zone in obs["target_temperature"]:
            target_temp = float(obs["target_temperature"][zone])
        if abs(current_temp - target_temp) > deadband_c:
            has_violation = True
            break

    comfort_penalty = violation_penalty if has_violation else 0.0
    total_reward = -(comfort_penalty + energy_weight * energy_penalty)

    return total_reward


@dataclass
class BarrierReward:
    controlled_zones: list[str]
    energy_weight: float
    deadband_c: float
    violation_penalty: float

    def __call__(self, obs):
        return barrier_reward_function(
            obs,
            self.controlled_zones,
            self.energy_weight,
            self.deadband_c,
            self.violation_penalty,
        )


def deadband_reward_function(
    obs,
    controlled_zones: list[str],
    energy_weight=1.0,
    target_temp: float = 21.0,
    dT: float = 0.5,
) -> float:
    # Energy observations are already in Wh/m² (converted in observation_spaces.py)
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

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
    controlled_zones: list[str]
    energy_weight: float
    target_temp: float
    dT: float

    def __call__(self, obs):
        return deadband_reward_function(
            obs=obs,
            controlled_zones=self.controlled_zones,
            energy_weight=self.energy_weight,
            target_temp=self.target_temp,
            dT=self.dT,
        )
