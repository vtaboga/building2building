from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class BaseRewardConfig:
    area: float


@dataclass
class BarrierRewardConfig:
    area: float


RewardConfig = Union[BaseRewardConfig, BarrierRewardConfig]


@dataclass
class BuildingConfig:
    path_to_building: Path
    path_to_weather: Path
    reward_config: RewardConfig
    energy_weight: float
    eplus_output_dir: Path
    warmup_phases: int
