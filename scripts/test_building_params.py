"""
Test script to verify building parameter extraction in the wrapper.
"""

import logging
from pathlib import Path
import tempfile

from building2building.sources import hydroquebec
from building2building.simulator import create_simulator
from building2building.simulator.wrappers import AugmentObservationWithBuildingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_building_param_extraction():
    """Test that building parameters are correctly extracted from environment metadata."""
    
    # Create a temporary directory for EnergyPlus outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        eplus_output_dir = Path(tmpdir)
        
        # Search for a building configuration
        logger.info("Searching for building configuration...")
        config_dict = {
            "env": {"control_mode": "hvac_actuators"},
            "bldg": {
                "geometry_unit_type": "single-family detached",
                "geometry_building_num_units": 1,
            }
        }
        
        configs = hydroquebec.search_configs(
            config=config_dict,
            n=1,
            eplus_output_dir=eplus_output_dir
        )
        
        if not configs:
            logger.error("No building configurations found!")
            return
        
        building_config = configs[0]
        logger.info(f"Found building config:")
        logger.info(f"  - Area: {building_config.area} m²")
        logger.info(f"  - Warmup phases: {building_config.warmup_phases}")
        logger.info(f"  - Num actuators: {len(building_config.hvac_actuators)}")
        
        # Create the environment
        logger.info("\nCreating environment...")
        env = create_simulator(building_config)
        
        # Check metadata
        logger.info("\nEnvironment metadata:")
        if hasattr(env, 'metadata'):
            logger.info(f"  - Area in metadata: {env.metadata.get('area', 'NOT FOUND')}")
            logger.info(f"  - Warmup phases in metadata: {env.metadata.get('warmup_phases', 'NOT FOUND')}")
            logger.info(f"  - Num actuators in metadata: {len(env.metadata.get('hvac_actuators', []))}")
        else:
            logger.error("  - No metadata found!")
        
        # Wrap with building parameter augmentation
        logger.info("\nWrapping environment with AugmentObservationWithBuildingParams...")
        wrapped_env = AugmentObservationWithBuildingParams(env)
        
        logger.info(f"\nExtracted building parameters:")
        logger.info(f"  - {wrapped_env.building_params}")
        
        logger.info(f"\nNormalized building parameters:")
        logger.info(f"  - {wrapped_env.normalized_params}")
        
        # Check observation space
        logger.info(f"\nOriginal observation space shape: {env.observation_space.shape}")
        logger.info(f"Augmented observation space shape: {wrapped_env.observation_space.shape}")
        
        expected_increase = len(wrapped_env.building_params)
        actual_increase = wrapped_env.observation_space.shape[0] - env.observation_space.shape[0]
        
        if actual_increase == expected_increase:
            logger.info(f"✓ Observation space correctly augmented by {actual_increase} dimensions")
        else:
            logger.error(f"✗ Expected increase of {expected_increase}, got {actual_increase}")
        
        # Test reset and observation augmentation
        logger.info("\nTesting environment reset and observation augmentation...")
        obs, info = wrapped_env.reset()
        logger.info(f"Augmented observation shape: {obs.shape}")
        logger.info(f"Last {len(wrapped_env.normalized_params)} values (building params): {obs[-len(wrapped_env.normalized_params):]}")
        
        logger.info("\n✓ Test completed successfully!")
        
        env.close()


if __name__ == "__main__":
    test_building_param_extraction()

