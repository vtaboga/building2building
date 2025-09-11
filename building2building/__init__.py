from pathlib import Path

from building2building.env import setup_energyplus_path
from building2building.sources import autobem

setup_energyplus_path()


from gymnasium.envs.registration import register

from building2building.simulator.create_simulator import create_simulator
from building2building.types import StateCode

register(
    id="EnergyPlus-v0",
    entry_point="building2building.simulator.create_simulator:create_simulator",
    kwargs={
        "building_config": None,  # Will be provided when creating env
    },
)


autobem_search_config = autobem.search_building_config


def autobem_env(
    state: StateCode,
    county: str,
    building_id: int | None = None,
    building_type: str | None = None,
    area: float | None = None,
    num_floors: int | None = None,
    height: float | None = None,
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
    config = autobem_search_config(
        state,
        county,
        building_id=building_id,
        building_type=building_type,
        area=area,
        num_floors=num_floors,
        height=height,
        eplus_output_dir=eplus_output_dir,
    )

    return create_simulator(config)


__all__ = [
    "autobem_search_config",
    "building_env",
]
