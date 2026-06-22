import itertools
import json
import logging
import warnings
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
    NormalizedDeadbandReward,
)
from building2building.geometry import extract_zone_geometry
from building2building.morphology import build_morphology
from building2building.types import (
    BuildingConfig,
    NormalizedDeadbandRewardConfig,
    TaskConfig,
)

logger = logging.getLogger(__name__)


# Calibration regime baked into reward_normalizers.yaml.
_CALIBRATION_DT: float = 1.0
_CALIBRATION_TARGET_MODE: str = "occupancy"

# Cap how many distinct (bt, bid, mode, dT) tuples we warn about per
# process.  Long PPO runs with many vec_env workers already deduplicate
# per process, but a multi-building sweep inside one process should not
# spam the log either.  The set is process-local; new worker processes
# emit again, which is the right behavior for SLURM array fan-out.
_NORMALIZED_REWARD_WARN_SEEN: set[tuple[str, str, str, float]] = set()


def _maybe_warn_normalized_deadband(
    *,
    task_config: TaskConfig,
    dT: float,
    building_type: str | None,
    building_id: str | None,
    tau_T: float,
    tau_E: float,
) -> None:
    """Emit ``RuntimeWarning`` (and ``logger.warning``) on calibration mismatch.

    Called once per simulator construction.  The check sees the
    *resolved* ``task_config.target_temperature_mode`` (which
    :func:`building2building.api.make_env` may have overridden via
    its ``target_temperature_mode=`` argument), so the warning matches
    the regime the env will actually run under, not whatever the preset
    originally said.

    Two independent conditions can fire (both, individually, or neither):

    * ``dT != 1.0``       — deadband shape differs from calibration.
    * ``mode != "occupancy"`` — target signal differs from calibration.

    Both proceed; the simulator is constructed with ``(tau_T, tau_E)``
    applied as-is.  This is intentional: the calibration is approximate
    outside the regime, and the cross-task benchmark in
    :mod:`building2building.benchmarks.goal_adaptation` quantifies the
    cost of that approximation.
    """
    bt = building_type or "<unknown>"
    bid = building_id or "<unknown>"
    mode = task_config.target_temperature_mode
    key = (bt, bid, mode, float(dT))
    if key in _NORMALIZED_REWARD_WARN_SEEN:
        return

    messages: list[str] = []
    if abs(dT - _CALIBRATION_DT) > 1e-9:
        messages.append(
            f"calibration regime mismatch: dT={dT!r} but reward_normalizers.yaml "
            f"was computed under dT={_CALIBRATION_DT!r}; "
            f"tau_T={tau_T:.6g} for ({bt}, {bid}) applied as-is."
        )
    if mode != _CALIBRATION_TARGET_MODE:
        messages.append(
            f"calibration regime mismatch: target_temperature_mode={mode!r} "
            f"but reward_normalizers.yaml was computed under "
            f"{_CALIBRATION_TARGET_MODE!r}; tau_T={tau_T:.6g}, "
            f"tau_E={tau_E:.6g} for ({bt}, {bid}) applied as-is."
        )

    if not messages:
        return

    _NORMALIZED_REWARD_WARN_SEEN.add(key)
    for msg in messages:
        # ``stacklevel=4`` so ``pytest.warns`` from a test that calls
        # ``make_env`` -> ``create_simulator`` ->
        # ``_maybe_warn_normalized_deadband`` reports the test frame
        # rather than this helper.
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
        # Also log so the warning shows up in logs/*.err under the
        # default Python warning filter, which can be set to "ignore"
        # by libraries we don't control.
        logger.warning(msg)


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

    controlled_zones = sorted(
        set(itertools.chain(*(item.zones() for item in building_config.hvac_equipment)))
    )

    heating_only_zones = sorted(
        set(
            z
            for eq in building_config.hvac_equipment
            if hasattr(eq, "equipment_type") and eq.equipment_type == "heating_only"
            for z in eq.zones()
        )
    )

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

    if not isinstance(building_config.reward_config, NormalizedDeadbandRewardConfig):
        raise ValueError(
            f"Unsupported reward type: {type(building_config.reward_config).__name__}. "
            "Only NormalizedDeadbandRewardConfig is supported."
        )
    cfg = building_config.reward_config
    if not cfg.is_filled:
        raise ValueError(
            "NormalizedDeadbandRewardConfig has unfilled tau_T/tau_E. "
            "This config is a preset sentinel; call "
            "`building2building.api.make_env(...)` (which auto-fills "
            "the constants from reward_normalizers.yaml) or explicitly "
            "call `cfg.filled(tau_T, tau_E)` before constructing the "
            "simulator."
        )
    # ``cfg.is_filled`` guarantees these are positive floats.
    assert cfg.tau_T is not None and cfg.tau_E is not None
    reward_function = NormalizedDeadbandReward(
        controlled_zones=reward_zones,
        energy_weight=cfg.energy_weight,
        dT=cfg.dT,
        tau_T=cfg.tau_T,
        tau_E=cfg.tau_E,
        task_config=task_config,
    )
    source_meta = (
        building_config.source_metadata
        if isinstance(building_config.source_metadata, dict)
        else {}
    )
    _maybe_warn_normalized_deadband(
        task_config=task_config,
        dT=cfg.dT,
        building_type=source_meta.get("building_type"),
        building_id=source_meta.get("building_id"),
        tau_T=cfg.tau_T,
        tau_E=cfg.tau_E,
    )

    # Finally, we compute the data necessary to fillin the metadata
    all_zones = set(str(z) for z in ont.zones())
    uncontrolled_zones = sorted(all_zones.difference(set(controlled_zones)))

    morphology = build_morphology(
        hvac_equipment=building_config.hvac_equipment,
        observation_names=obs_info.slot_names,
        action_names=action_names,
        controlled_zones=controlled_zones,
        all_zone_names=sorted(all_zones),
        zone_geometry=extract_zone_geometry(epjson),
    )

    gymenv = EnergyPlusEnvironment(
        make_energyplus,
        reward_function,
        obs_info.space,
        obs_info.flatten,
        action_space_info.agent_transform.codomain(),
        action_space_info.assemble_full_action,
        eplus_output_dir=eplus_output_dir,
        cleanup_output_dir_on_close=eplus_output_dir is not None,
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
        "building_source_metadata": (
            dict(building_config.source_metadata)
            if isinstance(building_config.source_metadata, dict)
            else {}
        ),
        "target_temperature_mode": task_config.target_temperature_mode,
        "morphology": morphology,
        "task_config": task_config,
        # ``building_info`` is populated by the env factories
        # (:func:`building2building.api.make_env`,
        # :func:`building2building.envs.factory.make_env_from_config`) after
        # simulator creation, since only they have the :class:`BuildingInfo`.
        "building_info": None,
    }

    return gymenv
