from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, Union


@dataclass
class BaseRewardConfig:
    energy_weight: float


@dataclass
class BarrierRewardConfig:
    energy_weight: float


@dataclass
class DeadbandRewardConfig:
    area: float
    energy_weight: float
    target_temp: float
    dT: float


RewardConfig = Union[DeadbandRewardConfig, BaseRewardConfig, BarrierRewardConfig]


@dataclass(frozen=True)
class ActuatorDescription:
    component_type: str
    control_type: str
    component_name: str
    units: str

    lower_bound: float
    upper_bound: float


class Equipment(Protocol):
    """This protocol is meant to describe what a 'controlled piece of equipment'
    provides: a set of actuators that we can control (some equipments will
    provide more than one) and a list of zones this actuator influences.

    """

    def actuator_descriptions(self) -> list[ActuatorDescription]: ...
    def zones(self) -> list[str]: ...


@dataclass
class BuildingConfig:
    path_to_building: Path
    path_to_weather: Path
    reward_config: RewardConfig
    eplus_output_dir: Path
    warmup_phases: int
    area: float
    hvac_equipment: Sequence[Equipment]
    # Optional metadata describing the building source/selection (e.g. dataset row id,
    # original IDF filename, weather station, etc.). This is meant for logging/debug.
    source_metadata: dict[str, Any] = field(default_factory=dict)
