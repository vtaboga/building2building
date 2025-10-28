from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class BaseRewardConfig:
    energy_weight: float


@dataclass
class BarrierRewardConfig:
    energy_weight: float


RewardConfig = Union[BaseRewardConfig, BarrierRewardConfig]


@dataclass
class BuildingConfig:
    path_to_building: Path
    path_to_weather: Path
    reward_config: RewardConfig
    eplus_output_dir: Path
    warmup_phases: int
    area: float
