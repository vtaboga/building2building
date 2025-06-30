import hydra
import logging
from omegaconf import DictConfig
from building2building.generator.search_idf import search_idf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for searching building files with Hydra configuration."""
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Extract building configuration
    building_cfg = cfg.building
    
    logger.info("Searching for building files...")
    search_result = search_idf(
        state=building_cfg.state,
        county=building_cfg.county,
        building_type=building_cfg.building_type,
        area=building_cfg.area,
        num_floors=building_cfg.num_floors,
        height=building_cfg.height,
        n_buildings=building_cfg.n_buildings,
        n_weather_files=building_cfg.get('n_weather_files', 2)
    )
    
    if search_result is None:
        logger.error("No buildings found matching the criteria")
        return
    
    buildings, path_to_weather = search_result
    
    logger.info(f"Found {len(buildings)} buildings")
    logger.info(f"Weather file path: {path_to_weather}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
