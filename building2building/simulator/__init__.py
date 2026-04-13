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

from building2building.simulator.action_spaces import (
    hvac_action_space,
)
from building2building.simulator.observation_spaces import (
    dict_observation_info,
    flat_observation_info,
)
from building2building.simulator.rewards import (
    BarrierReward,
    BaseReward,
    DeadbandReward,
)
from building2building.morphology import build_morphology
from building2building.types import (
    BarrierRewardConfig,
    BaseRewardConfig,
    BuildingConfig,
    DeadbandRewardConfig,
)


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
    max_steps: int = 200_000

    def __call__(self) -> EnergyPlusSimulation:
        sim = EnergyPlusSimulation(
            self.path_to_building,
            self.path_to_weather,
            self.observation_template,
            self.action_template,
            verbose=self.verbose,
            log_dir=self.log_dir,
            warmup_phases=self.warmup_phases,
            max_steps=self.max_steps,
        )

        return sim


def create_simulator(building_config: BuildingConfig) -> EnergyPlusEnvironment:
    """Create an EnergyPlus Gymnasium environment from a building config.

    Reads the epJSON building file, constructs observation and action spaces
    from the building's zones and HVAC equipment, selects the appropriate
    reward function, and returns a ready-to-use Gymnasium environment.

    Args:
        building_config: Complete building configuration including paths,
            reward settings, HVAC equipment, and task specification.

    Returns:
        A Gymnasium-compatible ``EnergyPlusEnvironment``.
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

    heating_only_zones = sorted(set(
        z
        for eq in building_config.hvac_equipment
        if hasattr(eq, "equipment_type") and eq.equipment_type == "heating_only"
        for z in eq.zones()
    ))

    task_config = building_config.task_config

    # We compute the observation side stuff
    obs_info = flat_observation_info(
        ont,
        area=building_config.area,
        controlled_zones=controlled_zones,
        task_config=task_config,
    )

    actuators = list(
        itertools.chain(
            *(item.actuator_descriptions() for item in building_config.hvac_equipment)
        )
    )

    if not building_config.expose_heating_only_zones:
        fixed_heating_only_names = frozenset(
            a.component_name
            for eq in building_config.hvac_equipment
            if hasattr(eq, "equipment_type") and eq.equipment_type == "heating_only"
            for a in eq.actuator_descriptions()
        )
    else:
        fixed_heating_only_names = frozenset()

    action_space_info = hvac_action_space(
        actuators,
        fixed_heating_only_names=fixed_heating_only_names,
        additional_fixed=building_config.fixed_actuator_overrides or None,
    )
    action_names = [
        f"{a.component_type}::{a.control_type}::{a.component_name}"
        for a in action_space_info.agent_actuators
    ]

    expected_steps = task_config.expected_steps()
    make_energyplus = MakeEnergyPlus(
        building_config.path_to_building,
        building_config.path_to_weather,
        obs_info.template,
        action_space_info.full_transform.domain(),
        verbose=False,
        log_dir=eplus_output_dir,
        warmup_phases=building_config.warmup_phases,
        max_steps=expected_steps + 1000,
    )

    reward_zones = (
        [z for z in controlled_zones if z not in set(heating_only_zones)]
        if not building_config.expose_heating_only_zones
        else controlled_zones
    )

    if isinstance(building_config.reward_config, BarrierRewardConfig):
        reward_function = BarrierReward(
            controlled_zones=reward_zones,
            energy_weight=building_config.reward_config.energy_weight,
            dT=building_config.reward_config.dT,
            violation_penalty=building_config.reward_config.violation_penalty,
            task_config=task_config,
        )
    elif isinstance(building_config.reward_config, BaseRewardConfig):
        reward_function = BaseReward(
            controlled_zones=reward_zones,
            energy_weight=building_config.reward_config.energy_weight,
            task_config=task_config,
        )
    elif isinstance(building_config.reward_config, DeadbandRewardConfig):
        reward_function = DeadbandReward(
            controlled_zones=reward_zones,
            energy_weight=building_config.reward_config.energy_weight,
            dT=building_config.reward_config.dT,
            task_config=task_config,
        )
    else:
        raise ValueError(f"Invalid reward type: {building_config.reward_config}")

    # Finally, we compute the data necessary to fillin the metadata
    all_zones = set(str(z) for z in ont.zones())
    uncontrolled_zones = sorted(all_zones.difference(set(controlled_zones)))

    morphology = build_morphology(
        hvac_equipment=building_config.hvac_equipment,
        observation_names=obs_info.slot_names,
        action_names=action_names,
        controlled_zones=controlled_zones,
        all_zone_names=sorted(all_zones),
    )

    gymenv = EnergyPlusEnvironment[np.ndarray, np.ndarray](
        make_energyplus,
        reward_function,
        obs_info.space,
        obs_info.flatten,
        action_space_info.agent_transform.codomain(),
        action_space_info.assemble_full_action,
    )

    gymenv.metadata = {
        "controlled_zones": controlled_zones,
        "heating_only_zones": heating_only_zones,
        "uncontrolled_zones": uncontrolled_zones,
        "observation_names": obs_info.slot_names,
        "action_names": action_names,
        "hvac_equipment": building_config.hvac_equipment,
        "area": building_config.area,
        "warmup_phases": building_config.warmup_phases,
        "building_source_metadata": dict(building_config.source_metadata)
        if isinstance(building_config.source_metadata, dict)
        else {},
        "target_temperature_mode": task_config.target_temperature_mode,
        "morphology": morphology,
    }

    return gymenv
