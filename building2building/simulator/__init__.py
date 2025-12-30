import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from minergym.environment import EnergyPlusEnvironment
from minergym.ontology import Ontology
from minergym.simulation import EnergyPlusSimulation

from building2building.simulator.action_spaces import (
    get_controllable_setpoints,
    hvac_actuators_transform,
    hvac_actuators_multidiscrete_transform,
    many_thermostats_transform,
    many_thermostats_transform_dict,
)
from building2building.simulator.observation_spaces import (
    dict_observation_info,
    flat_observation_info,
)
from building2building.simulator.rewards import (
    BarrierReward,
    BaseReward,
    DeadbandReward
)
from building2building.types import (
    DeadbandRewardConfig,
    BarrierRewardConfig,
    BaseRewardConfig,
    BuildingConfig,
)

from .transform_utils import TransformInverse

logger = logging.getLogger(__name__)


@dataclass
class MakeEnergyPlus:
    """This could simply be a closure, but it wouldn't be serializable with
    pickle."""

    path_to_building: Path
    path_to_weather: Path
    observation_template: Any
    action_template: Any
    verbose: bool
    log_dir: Path
    warmup_phases: int

    def __call__(self) -> EnergyPlusSimulation:
        sim = EnergyPlusSimulation(
            self.path_to_building,
            self.path_to_weather,
            self.observation_template,
            self.action_template,
            verbose=self.verbose,
            log_dir=self.log_dir,
            warmup_phases=self.warmup_phases,
        )

        return sim


def create_simulator(building_config: BuildingConfig) -> EnergyPlusEnvironment:
    """
    Create a simulator for a given building and weather file.

    Args:
        path_to_building: Path to the building file (epJSON)
        path_to_weather: Path to the weather file (epw)
        building_characteristics: Dictionary with building characteristics
        reward_type: Type of reward function to use
        energy_weight: Energy weight for reward calculation
        eplus_output_dir: Optional directory for EnergyPlus outputs

    Returns:
        gym.Env: EnergyPlus environment
    """

    if not isinstance(building_config, BuildingConfig):
        # If the type constraints are satisfied, it should be unreachable, but
        # this function is called through gymnasium.make, which doesn't
        # propagate type constraints.
        raise Exception(f"{building_config} should have type BuildingConfig")

    eplus_output_dir = building_config.eplus_output_dir

    ont = Ontology.from_json(building_config.path_to_building)

    # We compute the observation side stuff
    obs_info = flat_observation_info(ont, area=building_config.area)

    # Then the action side stuff
    setpoints = get_controllable_setpoints(ont)

    if building_config.hvac_actuators:
        if building_config.hvac_action_space == "multidiscrete":
            action_transform = hvac_actuators_multidiscrete_transform(
                building_config.hvac_actuators,
                n_bins_continuous=int(building_config.n_bins_continuous),
            )
        else:
            action_transform = hvac_actuators_transform(building_config.hvac_actuators)
        action_names = [
            f"{a['component_type']}::{a['control_type']}::{a['component_name']}"
            for a in building_config.hvac_actuators
        ]
    else:
        thermostat_list = list(itertools.chain(*setpoints.values()))
        action_transform = many_thermostats_transform(thermostat_list)
        action_names = []

    make_energyplus = MakeEnergyPlus(
        building_config.path_to_building,
        building_config.path_to_weather,
        obs_info.template,
        action_transform.domain(),
        verbose=False,
        log_dir=eplus_output_dir,
        warmup_phases=building_config.warmup_phases,
    )

    if isinstance(building_config.reward_config, BarrierRewardConfig):
        reward_function = BarrierReward(
            area=building_config.area,
            setpoints=setpoints,
            energy_weight=building_config.reward_config.energy_weight,
        )
    elif isinstance(building_config.reward_config, BaseRewardConfig):
        reward_function = BaseReward(
            area=building_config.area,
            setpoints=setpoints,
            energy_weight=building_config.reward_config.energy_weight,
        )
    elif isinstance(building_config.reward_config, DeadbandRewardConfig):
        reward_function = DeadbandReward(
            area=building_config.area,
            setpoints=setpoints,
            energy_weight=building_config.reward_config.energy_weight,
            target_temp=building_config.reward_config.target_temp,
            dT=building_config.reward_config.dT,
        )
    else:
        raise ValueError(f"Invalid reward type: {building_config.reward_config}")

    # Finally, we compute the data necessary to fillin the metadata
    controlled_zones = list(setpoints.keys())
    all_zones = [z.toPython() for z in ont.zones()]
    uncontrolled_zones = [zone for zone in all_zones if zone not in controlled_zones]

    gymenv = EnergyPlusEnvironment[np.ndarray, np.ndarray](
        make_energyplus,
        reward_function,
        obs_info.space,
        obs_info.flatten,
        action_transform.codomain(),
        TransformInverse(action_transform),
    )

    gymenv.metadata = {
        "controlled_zones": controlled_zones,
        "uncontrolled_zones": uncontrolled_zones,
        "observation_names": obs_info.slot_names,
        "action_names": action_names,
        "hvac_actuators": building_config.hvac_actuators,
    }

    return gymenv


def create_simulator_dict(building_config: BuildingConfig) -> EnergyPlusEnvironment:
    if not isinstance(building_config, BuildingConfig):
        # If the type constraints are satisfied, it should be unreachable, but
        # this function is called through gymnasium.make, which doesn't
        # propagate type constraints.
        raise Exception(f"{building_config} should have type BuildingConfig")

    eplus_output_dir = building_config.eplus_output_dir

    ont = Ontology.from_json(building_config.path_to_building)

    # We compute the observation side stuff
    obs_info = dict_observation_info(ont, area=building_config.area)

    # Then the action side stuff
    setpoints = get_controllable_setpoints(ont)

    if building_config.hvac_actuators:
        # Dict-mode action spaces for HVAC actuators are not implemented yet.
        # The existing tests and scripts use the flat action space.
        raise NotImplementedError(
            "HVAC actuator control mode is only implemented for the flat action space."
        )

    thermostat_list = list(itertools.chain(*setpoints.values()))
    action_transform = many_thermostats_transform_dict(thermostat_list)

    make_energyplus = MakeEnergyPlus(
        building_config.path_to_building,
        building_config.path_to_weather,
        obs_info.domain(),
        action_transform.domain(),
        verbose=False,
        log_dir=eplus_output_dir,
        warmup_phases=building_config.warmup_phases,
    )

    if isinstance(building_config.reward_config, BarrierRewardConfig):
        reward_function = BarrierReward(
            building_config.area,
            setpoints,
            building_config.reward_config.energy_weight,
        )
    elif isinstance(building_config.reward_config, BaseRewardConfig):
        reward_function = BaseReward(
            building_config.area,
            setpoints,
            building_config.reward_config.energy_weight,
        )
    elif isinstance(building_config.reward_config, DeadbandRewardConfig):
        reward_function = DeadbandReward(
            area=building_config.area,
            setpoints=setpoints,
            energy_weight=building_config.reward_config.energy_weight,
            target_temp=building_config.reward_config.target_temp,
            dT=building_config.reward_config.dT,
        )
    else:
        raise ValueError(f"Invalid reward type: {building_config.reward_config}")

    # Finally, we compute the data necessary to fill in the metadata
    controlled_zones = list(setpoints.keys())
    all_zones = [z.toPython() for z in ont.zones()]
    uncontrolled_zones = [zone for zone in all_zones if zone not in controlled_zones]

    gymenv = EnergyPlusEnvironment[np.ndarray, np.ndarray](
        make_energyplus,
        reward_function,
        obs_info.codomain(),
        obs_info,
        action_transform.codomain(),
        TransformInverse(action_transform),
    )

    gymenv.metadata = {
        "controlled_zones": controlled_zones,
        "uncontrolled_zones": uncontrolled_zones,
    }

    return gymenv
