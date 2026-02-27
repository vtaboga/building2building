from dataclasses import dataclass
from typing import Any

import numpy as np

from b2b.types import TaskConfig


def _zone_target(obs: dict[str, Any], zone: str, task_config: TaskConfig) -> float:
    """Return the target temperature for *zone*.

    When the observation contains a dynamic ``target_temperature`` group
    (occupancy mode), read it from there.  Otherwise fall back to the
    constant target stored in *task_config*.
    """
    target_temps = obs.get("target_temperature", {})
    if zone in target_temps:
        return float(target_temps[zone])
    return task_config.target_for_zone(zone).occupied_c


def base_reward_function(
    obs: dict[str, Any],
    controlled_zones: list[str],
    task_config: TaskConfig,
    energy_weight: float = 1.0,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Args:
        obs: Dictionary containing observations (energy values in Wh/m²)
        controlled_zones: List of controlled zone names
        task_config: Task configuration with target temperature info
        energy_weight: Weight for the energy consumption penalty
    Returns:
        float: Combined reward (negative values represent penalties)
    """

    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    temp_error = 0.0
    for zone in controlled_zones:
        current_temp = float(obs["temperature"][zone])
        target_temp = _zone_target(obs, zone, task_config)
        temp_error += (current_temp - target_temp) ** 2

    temp_error = temp_error / len(controlled_zones)

    total_reward = -(temp_error + energy_weight * energy_penalty)

    return total_reward


@dataclass
class BaseReward:
    controlled_zones: list[str]
    energy_weight: float
    task_config: TaskConfig

    def __call__(self, obs: dict[str, Any]) -> float:
        return base_reward_function(
            obs, self.controlled_zones, self.task_config, self.energy_weight
        )


def barrier_reward_function(
    obs: dict[str, Any],
    controlled_zones: list[str],
    task_config: TaskConfig,
    energy_weight: float = 1.0,
    deadband_c: float = 0.5,
    violation_penalty: float = 100.0,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Args:
        obs: Dictionary containing observations (energy values in Wh/m²)
        controlled_zones: List of controlled zone names
        task_config: Task configuration with target temperature info
        energy_weight: Weight for the energy consumption penalty
        deadband_c: Comfort deadband in °C
        violation_penalty: Penalty for comfort violations

    Returns:
        float: Combined reward (negative values represent penalties)
    """

    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    comfort_penalty = 0.0
    for zone in controlled_zones:
        current_temp = float(obs["temperature"][zone])
        target_temp = _zone_target(obs, zone, task_config)
        deviation = abs(current_temp - target_temp)
        if deviation > deadband_c:
            comfort_penalty += violation_penalty * (deviation - deadband_c)
        else:
            comfort_penalty += (current_temp - target_temp) ** 2

    comfort_penalty = comfort_penalty / len(controlled_zones)
    total_reward = -(comfort_penalty + energy_weight * energy_penalty)

    return total_reward


@dataclass
class BarrierReward:
    controlled_zones: list[str]
    energy_weight: float
    deadband_c: float
    violation_penalty: float
    task_config: TaskConfig

    def __call__(self, obs: dict[str, Any]) -> float:
        return barrier_reward_function(
            obs,
            self.controlled_zones,
            self.task_config,
            self.energy_weight,
            self.deadband_c,
            self.violation_penalty,
        )


def deadband_reward_function(
    obs: dict[str, Any],
    controlled_zones: list[str],
    task_config: TaskConfig,
    energy_weight: float = 1.0,
    dT: float = 0.5,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Inside deadband (|T - target| <= dT):  -(T - target)^2
    Outside deadband (|T - target| > dT):  -|T - target|

    The quadratic term in the deadband avoid bang-bang behavior.
    """
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    temp_error = 0.0
    for zone in controlled_zones:
        current_temp = float(obs["temperature"][zone])
        target_temp = _zone_target(obs, zone, task_config)
        dev = abs(current_temp - target_temp)
        if dev <= dT:
            temp_error += (current_temp - target_temp) ** 2
        else:
            temp_error += -dev

    temp_error = temp_error / len(controlled_zones)

    total_reward = -(temp_error + energy_weight * energy_penalty)

    return total_reward


@dataclass
class DeadbandReward:
    controlled_zones: list[str]
    energy_weight: float
    dT: float
    task_config: TaskConfig

    def __call__(self, obs: dict[str, Any]) -> float:
        return deadband_reward_function(
            obs=obs,
            controlled_zones=self.controlled_zones,
            task_config=self.task_config,
            energy_weight=self.energy_weight,
            dT=self.dT,
        )
