"""Reward functions for HVAC control environments.

Provides the normalized deadband reward used across all task presets:

* :class:`NormalizedDeadbandReward` -- deadband reward with
  per-(building_type, climate_zone) ``(tau_T, tau_E)`` normalizers so
  that ``energy_weight`` is dimensionless and comparable across
  buildings.
"""

from dataclasses import dataclass
from typing import Any

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


def _deadband_components(
    obs: dict[str, Any],
    controlled_zones: list[str],
    task_config: TaskConfig,
    dT: float,
) -> tuple[float, float]:
    """Compute the deadband ``(temp_penalty, power_penalty)`` decomposition."""
    
    energy_penalty = obs["energy"]["electricity"] + obs["energy"]["natural_gas"]

    temp_error = 0.0
    for zone in controlled_zones:
        current_temp = float(obs["temperature"][zone])
        target_temp = _zone_target(obs, zone, task_config)
        temp_error += (current_temp - target_temp) ** 2

    temp_error = temp_error / len(controlled_zones)

    return float(temp_error), float(energy_penalty)


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

    Computes the ``(temp_penalty, power_penalty)`` decomposition via
    :func:`_deadband_components`, then returns

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
