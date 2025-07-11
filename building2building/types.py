import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self


@dataclass
class BuildingCharacteristics:
    building_type: str
    num_floors: int
    area: float
    height: float
    zone_lists: list[str]

    @classmethod
    def load_json(cls, json_path: Path) -> Self:
        """Load building characteristics from a JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_json(self, json_path: Path):
        with open(json_path, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create instance from dictionary."""
        return cls(
            building_type=data["building_type"],
            num_floors=data["num_floors"],
            area=data["area"],
            height=data["height"],
            zone_lists=data["zone_lists"],
        )


RewardType = Literal["barrier", "base"]


@dataclass
class BuildingConfig:
    path_to_building: Path
    path_to_weather: Path
    characteristics: BuildingCharacteristics
    reward_type: RewardType
    energy_weight: float
    eplus_output_dir: Path
