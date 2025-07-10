from pathlib import Path

from building2building.env import setup_energyplus_path
from building2building.generator.downloader import download_epw
from building2building.generator.search_idf import EPJSONProcessor, search_idf
from building2building.types import BuildingConfig

setup_energyplus_path()

search_building = search_idf


def search_weather(state: str, n_files: int) -> list[Path]:
    # Download associated weather files
    weather_files = download_epw(state_code=state, n_files=n_files)
    return weather_files


def building_config(
    state: str,
    county: str,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
    processors: list[EPJSONProcessor] = [],
    eplus_output_dir: Path = Path("eplus_out"),
) -> BuildingConfig:
    weather_files = search_weather(state, 1)
    if len(weather_files) == 0:
        raise Exception("No such weathers")
    building_files = search_building(
        state,
        county,
        1,
        building_type=building_type,
        area=area,
        num_floors=num_floors,
        height=height,
        processors=processors,
    )
    if len(building_files) == 0:
        raise Exception("No such buildings")
    building_path, characteristics = building_files[0]

    return BuildingConfig(
        path_to_building=building_path,
        path_to_weather=weather_files[0],
        characteristics=characteristics,
        reward_type="base",
        energy_weight=1.0,
        eplus_output_dir=eplus_output_dir,
    )


__all__ = [
    "search_building",
    "search_weather",
    "building_config",
]
