from pathlib import Path

from gymnasium.envs.registration import register

from building2building.env import DataPaths, setup_energyplus_path
from building2building.generator.downloader import download_epw
from building2building.generator.search_idf import EPJSONProcessor, search_idf
from building2building.simulator.create_simulator import create_simulator
from building2building.types import BuildingConfig, RewardType, StateCode

register(
    id="EnergyPlus-v0",
    entry_point="building2building.simulator.create_simulator:create_simulator",
    kwargs={
        "building_config": None,  # Will be provided when creating env
    },
)


setup_energyplus_path()

DataPaths.setup()

search_building = search_idf


def search_weather(state: StateCode, n_files: int) -> list[Path]:
    """
    Search and download weather files for a given state.

    Args:
        state: The state code to search weather files for
        n_files: The number of weather files to download

    Returns:
        A list of Path objects pointing to the downloaded weather files
    """
    weather_files = download_epw(state_code=state, n_files=n_files)
    return weather_files


def building_configs(
    state: StateCode,
    county: str,
    building_id: int | None = None,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
    processors: list[EPJSONProcessor] = [],
    eplus_output_dir: Path = Path("eplus_out"),
    reward_type: RewardType = "base",
    *,
    n_buildings,
    n_weathers,
) -> list[BuildingConfig]:
    weather_files = search_weather(state, n_weathers)
    if len(weather_files) == 0:
        raise Exception("No such weathers")
    building_files = search_idf(
        state,
        county,
        n_buildings,
        building_id=building_id,
        building_type=building_type,
        area=area,
        num_floors=num_floors,
        height=height,
        processors=processors,
    )
    if len(building_files) == 0:
        raise Exception("No such buildings")

    return [
        BuildingConfig(
            path_to_building=building_file,
            path_to_weather=weather_file,
            characteristics=characteristics,
            reward_type=reward_type,
            energy_weight=1.0,
            eplus_output_dir=eplus_output_dir,
        )
        for (building_file, characteristics) in building_files
        for weather_file in weather_files
    ]


def building_config(
    state: StateCode,
    county: str,
    building_id: int | None = None,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
    processors: list[EPJSONProcessor] = [],
    eplus_output_dir: Path = Path("eplus_out"),
    reward_type: RewardType = "base",
) -> BuildingConfig:
    """Create a building configuration by searching for weather and building files.

    Searches for weather files in the specified state and building files in the
    specified state and county. Returns a BuildingConfig object with the first
    matching weather and building files found.

    Args:
        state: The state to search for weather and building files
        county: The county to search for building files
        building_type: Optional building type filter for building search
        area: Optional area filter for building search
        num_floors: Optional number of floors filter for building search
        height: Optional height filter for building search
        processors: List of EPJSONProcessor objects to apply to building
        eplus_output_dir: Directory path for EnergyPlus output files
        reward_type: Type of reward calculation to use

    Returns:
        BuildingConfig object configured with found weather and building files
    """
    weather_files = search_weather(state, 1)
    if len(weather_files) == 0:
        raise Exception("No such weathers")
    building_files = search_building(
        state,
        county,
        1,
        building_id=building_id,
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
        reward_type=reward_type,
        energy_weight=1.0,
        eplus_output_dir=eplus_output_dir,
    )


def building_env(
    state: StateCode,
    county: str,
    building_id: int | None = None,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
    processors: list[EPJSONProcessor] = [],
    eplus_output_dir: Path = Path("eplus_out"),
):
    """Create a building environment simulator from the first results that
    would be returned by `building_config`.

    Args:
        state: The state where the building is located
        county: The county where the building is located
        building_type: Optional type of building to simulate
        area: Optional floor area of the building in square units
        num_floors: Optional number of floors in the building
        height: Optional height of the building
        processors: List of EPJSON processors to apply to the building configuration
        eplus_output_dir: Directory path for EnergyPlus output files

    Returns:
        A configured building simulator environment

    """
    config = building_config(
        state,
        county,
        building_id=building_id,
        building_type=building_type,
        area=area,
        num_floors=num_floors,
        height=height,
        processors=processors,
        eplus_output_dir=eplus_output_dir,
    )

    return create_simulator(config)


__all__ = [
    "search_building",
    "search_weather",
    "building_config",
    "building_configs",
    "building_env",
]
