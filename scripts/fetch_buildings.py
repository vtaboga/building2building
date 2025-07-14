import logging

import hydra
from building2building import search_weather
from building2building.generator.search_idf import search_idf
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Main function for searching building files with Hydra configuration."""

    # Extract building configuration
    building_cfg = cfg.building

    logger.info("Searching for building files...")

    buildings = search_idf(
        state=building_cfg.state,
        county=building_cfg.county,
        building_type=building_cfg.building_type,
        area=building_cfg.area,
        num_floors=building_cfg.num_floors,
        height=building_cfg.height,
        n_buildings=building_cfg.n_buildings,
    )

    weathers = search_weather(
        building_cfg.state,
        n_files=1,
    )

    if len(buildings) == 0:
        logger.error("No buildings found matching the criteria")
        return

    logger.info(f"Found {len(buildings)} buildings")
    logger.info(f"Weather file path: {weathers[0]}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
