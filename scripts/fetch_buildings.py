import hydra
from omegaconf import DictConfig
from building2building.generator.search_idf import search_idf
from building2building.core.hydra_manager import HydraManager


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Fetch buildings using Hydra configuration."""
    
    # Create a Hydra-compatible run manager
    run_manager = HydraManager(cfg)
    logger = run_manager.logger
    
    # Extract building configuration
    building_cfg = cfg.building
    
    logger.info("Searching for building files...")
    buildings, path_to_weather = search_idf(
        state=building_cfg.state,
        county=building_cfg.county,
        building_type=building_cfg.building_type,
        area=building_cfg.area,
        num_floors=building_cfg.num_floors,
        height=building_cfg.height,
        n_buildings=building_cfg.n_buildings,
        n_weather_files=building_cfg.get('n_weather_files', 2)
    )
    
    logger.info(f"Found {len(buildings)} buildings")
    logger.info(f"Weather file path: {path_to_weather}")
    logger.info("Done!")
    
    # Finalize the run
    run_manager.finish()


if __name__ == "__main__":
    main()
