import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from minergym.environment import EnergyPlusEnvironment
from minergym.ontology import Ontology
from minergym.simulation import EnergyPlusSimulation

from b2b.simulator.action_spaces import (
    hvac_actuators_transform,
)
from b2b.simulator.observation_spaces import (
    dict_observation_info,
    flat_observation_info,
)
from b2b.simulator.rewards import (
    BarrierReward,
    BaseReward,
    DeadbandReward,
)
from b2b.types import (
    BarrierRewardConfig,
    BaseRewardConfig,
    BuildingConfig,
    DeadbandRewardConfig,
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

    with open(building_config.path_to_building, "r") as epjson_file:
        epjson: dict[str, Any] = json.load(epjson_file)

    ont = Ontology.from_object(epjson)

    controlled_zones = list(
        set(itertools.chain(*(item.zones() for item in building_config.hvac_equipment)))
    )

    # We compute the observation side stuff
    obs_info = flat_observation_info(ont, area=building_config.area)

    actuators = list(
        itertools.chain(
            *(item.actuator_descriptions() for item in building_config.hvac_equipment)
        )
    )

    # Then the action side stuff
    # TODO: this is used for the reward functions, but come from actuatorsv1
    # setpoints = get_controllable_setpoints(ont)

    action_transform = hvac_actuators_transform(actuators)
    action_names = [
        f"{a.component_type}::{a.control_type}::{a.component_name}" for a in actuators
    ]

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
            controlled_zones=controlled_zones,
            energy_weight=building_config.reward_config.energy_weight,
        )
    elif isinstance(building_config.reward_config, BaseRewardConfig):
        reward_function = BaseReward(
            area=building_config.area,
            controlled_zones=controlled_zones,
            energy_weight=building_config.reward_config.energy_weight,
        )
    elif isinstance(building_config.reward_config, DeadbandRewardConfig):
        reward_function = DeadbandReward(
            area=building_config.area,
            controlled_zones=controlled_zones,
            energy_weight=building_config.reward_config.energy_weight,
            target_temp=building_config.reward_config.target_temp,
            dT=building_config.reward_config.dT,
        )
    else:
        raise ValueError(f"Invalid reward type: {building_config.reward_config}")

    # Finally, we compute the data necessary to fillin the metadata
    all_zones = set(str(z) for z in ont.zones())
    uncontrolled_zones = sorted(all_zones.difference(set(controlled_zones)))

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
        "area": building_config.area,
        "warmup_phases": building_config.warmup_phases,
        "building_source_metadata": dict(building_config.source_metadata)
        if isinstance(building_config.source_metadata, dict)
        else {},
    }

    return gymenv
