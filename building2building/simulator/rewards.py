"""Reward functions for HVAC control environments.

Provides four reward variants that trade off thermal comfort against
energy consumption:

* :class:`BaseReward` -- MSE temperature tracking + weighted energy penalty.
* :class:`BarrierReward` -- deadband with steep violation penalty.
* :class:`DeadbandReward` -- quadratic inside deadband, linear outside.
* :class:`NormalizedDeadbandReward` -- :class:`DeadbandReward` with
  per-(building_type, climate_zone) ``(tau_T, tau_E)`` normalizers so
  that ``energy_weight`` is dimensionless and comparable across
  buildings.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from building2building.types import TaskConfig


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
    dT: float = 1.0,
    violation_penalty: float = 10.0,
) -> float:
    """Calculate a reward combining temperature tracking and energy consumption.

    Args:
        obs: Dictionary containing observations (energy values in Wh/m²)
        controlled_zones: List of controlled zone names
        task_config: Task configuration with target temperature info
        energy_weight: Weight for the energy consumption penalty
        dT: Comfort deadband in °C
        violation_penalty: Penalty for comfort violations

    Returns:
        float: Combined reward (negative values represent penalties)
    """

    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    temp_error = 0.0
    for zone in controlled_zones:
        current_temp = float(obs["temperature"][zone])
        target_temp = _zone_target(obs, zone, task_config)
        dev = abs(current_temp - target_temp)
        if dev >= dT:
            temp_error += violation_penalty * (dev + 1)

    temp_error = temp_error / len(controlled_zones)
    total_reward = -(temp_error + energy_weight * energy_penalty)

    return total_reward


@dataclass
class BarrierReward:
    controlled_zones: list[str]
    energy_weight: float
    dT: float
    violation_penalty: float
    task_config: TaskConfig

    def __call__(self, obs: dict[str, Any]) -> float:
        return barrier_reward_function(
            obs,
            self.controlled_zones,
            self.task_config,
            self.energy_weight,
            self.dT,
            self.violation_penalty,
        )


def _deadband_components(
    obs: dict[str, Any],
    controlled_zones: list[str],
    task_config: TaskConfig,
    dT: float,
) -> tuple[float, float]:
    """Compute the deadband ``(temp_penalty, power_penalty)`` decomposition.

    ``temp_penalty`` is the per-zone-averaged deadband distance:
    quadratic for ``|T - target| <= dT`` and linear (in ``|T - target|``)
    beyond.  ``power_penalty`` is electricity + natural-gas energy in
    Wh/m².  Both are *non-negative*; the reward sign flip happens in
    the callers.

    Sharing this helper between :class:`DeadbandReward` and
    :class:`NormalizedDeadbandReward` makes the ``tau=1`` equivalence
    test mechanical: at ``tau_T = tau_E = 1.0`` and the same
    ``energy_weight`` and ``dT``, the two rewards must produce
    identical outputs.
    """
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    temp_error = 0.0
    for zone in controlled_zones:
        current_temp = float(obs["temperature"][zone])
        target_temp = _zone_target(obs, zone, task_config)
        temp_error += (current_temp - target_temp) ** 2

    temp_error = temp_error / len(controlled_zones)

    return float(temp_error), float(energy_penalty)


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
    temp_penalty, power_penalty = _deadband_components(
        obs, controlled_zones, task_config, dT
    )
    return -(temp_penalty + energy_weight * power_penalty)


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


def normalized_deadband_reward_function(
    obs: dict[str, Any],
    controlled_zones: list[str],
    task_config: TaskConfig,
    energy_weight: float,
    dT: float,
    tau_T: float,
    tau_E: float,
) -> float:
    """Per-bucket-normalized version of :func:`deadband_reward_function`.

    Computes the same ``(temp_penalty, power_penalty)`` decomposition
    as :class:`DeadbandReward`, then returns

    .. math::

        r = -\\Big(\\tfrac{\\text{temp\\_penalty}}{\\tau_T}
                  + w_E \\cdot \\tfrac{\\text{power\\_penalty}}{\\tau_E}\\Big).

    See
    :class:`building2building.types.NormalizedDeadbandRewardConfig`
    for the rationale and calibration regime.
    """
    temp_penalty, power_penalty = _deadband_components(
        obs, controlled_zones, task_config, dT
    )
    return -(temp_penalty / tau_T + energy_weight * power_penalty / tau_E)


@dataclass
class NormalizedDeadbandReward:
    """Deadband reward with per-bucket ``(tau_T, tau_E)`` normalizers.

    The dispatch site in :mod:`building2building.simulator` is
    responsible for resolving the ``(tau_T, tau_E)`` for the building
    being simulated and rejecting unfilled
    :class:`~building2building.types.NormalizedDeadbandRewardConfig`
    sentinels, so by the time this object is constructed both values
    are positive floats.
    """

    controlled_zones: list[str]
    energy_weight: float
    dT: float
    tau_T: float
    tau_E: float
    task_config: TaskConfig

    def __call__(self, obs: dict[str, Any]) -> float:
        return normalized_deadband_reward_function(
            obs=obs,
            controlled_zones=self.controlled_zones,
            task_config=self.task_config,
            energy_weight=self.energy_weight,
            dT=self.dT,
            tau_T=self.tau_T,
            tau_E=self.tau_E,
        )
