import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import logging
import json
import os
from pathlib import Path
from building2building.algorithms.online.baselines import run_constant_baseline

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    """Run constant baseline policy on EnergyPlus environment"""
    
    # Get Hydra's output directory (automatically managed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    
    # Create EnergyPlus output directory within the run directory
    eplus_output_dir = output_dir / "eplus_output"
    
    # Setup paths
    building_path = f"data/processed_buildings/{cfg.state}/{cfg.county}/{cfg.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{cfg.state}/{cfg.county}/{cfg.building_id}.json"
    weather_path = f"data/weather/{cfg.weather}"
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        logger.error(f"Building characteristics file not found: {characteristics_path}")
        return
    
    # Run the constant baseline evaluation using Hydra's output directory
    results = run_constant_baseline(
        env_id="EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=weather_path,
        building_characteristics=building_characteristics,
        heating_setpoint=cfg.heating_setpoint,
        cooling_setpoint=cfg.cooling_setpoint,
        reward_type=cfg.reward_type,
        energy_weight=cfg.energy_weight,
        seed=cfg.seed,
        results_dir=str(output_dir),
        eplus_output_dir=str(eplus_output_dir)
    )
    
    logger.info("Baseline Evaluation Results:")
    logger.info(f"Total reward: {results['total_reward']:.2f}")
    logger.info(f"Mean reward per step: {results['mean_reward']:.2f}")
    logger.info(f"Total timesteps: {results['total_timesteps']}")
    logger.info(f"Results saved to: {output_dir / 'baseline_results.json'}")
    logger.info(f"EnergyPlus output saved to: {eplus_output_dir}")
    logger.info(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()

