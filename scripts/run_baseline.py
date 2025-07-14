import json
import logging
import os
from pathlib import Path

import hydra
from building2building.algorithms.online.baselines import run_constant_baseline
from building2building.env import DataPaths
from building2building.types import BuildingCharacteristics, BuildingConfig
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    """Run constant baseline policy on EnergyPlus environment"""

    # Get Hydra's output directory (automatically managed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Create EnergyPlus output directory within the run directory
    eplus_output_dir = output_dir / "eplus_output"

    # Setup paths

    building_path = (
        DataPaths.processed_dir() / cfg.state / cfg.county / f"{cfg.building_id}.epJSON"
    )
    characteristics_path = (
        DataPaths.processed_dir() / cfg.state / cfg.county / f"{cfg.building_id}.json"
    )
    weather_path: Path = (
        DataPaths.weather_dir() / cfg.weather_validation
    )  # No training, apply policy to validation weather

    try:
        building_characteristics = BuildingCharacteristics.load_json(
            characteristics_path
        )
    except FileNotFoundError:
        logger.error(f"Building characteristics file not found: {characteristics_path}")
        return

    building_config = BuildingConfig(
        building_path,
        weather_path,
        building_characteristics,
        cfg.reward_type,
        cfg.energy_weight,
        eplus_output_dir,
    )
    # Run the constant baseline evaluation using Hydra's output directory
    results = run_constant_baseline(
        env_id="EnergyPlus-v0",
        building_config=building_config,
        heating_setpoint=cfg.constant.heating_setpoint,
        cooling_setpoint=cfg.constant.cooling_setpoint,
        seed=cfg.seed,
        results_dir=output_dir,
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
