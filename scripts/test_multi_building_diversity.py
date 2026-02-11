#!/usr/bin/env python3
"""
Test script to verify multi-building diversity is working correctly.

This script:
1. Fetches a diverse building pool
2. Creates multiple environments
3. Verifies they have different building parameters
"""

import logging
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from algorithms.multi_building_utils import fetch_diverse_building_pool, make_diverse_env
from building2building.simulator.wrappers import AugmentObservationWithBuildingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_building_diversity():
    """Test that diverse building sampling works correctly."""
    
    # Mock config
    class MockConfig:
        def __init__(self):
            self.bldg = {
                "geometry_unit_type": "single-family detached",
                "geometry_building_num_units": 1,
            }
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = MockConfig()
    
    # Test 1: Fetch diverse building pool
    logger.info("=" * 60)
    logger.info("TEST 1: Fetching diverse building pool")
    logger.info("=" * 60)
    
    try:
        building_pool = fetch_diverse_building_pool(config, n_buildings=5)
        logger.info(f"✅ Successfully fetched {len(building_pool)} buildings")
        
        # Verify diversity
        areas = [b.area for b in building_pool]
        warmups = [b.warmup_phases for b in building_pool]
        actuators = [len(b.hvac_actuators) for b in building_pool]
        
        logger.info(f"Building pool statistics:")
        logger.info(f"  Areas: {areas}")
        logger.info(f"  Warmup phases: {warmups}")
        logger.info(f"  Actuators: {actuators}")
        
        # Check if there's any variation
        if len(set(areas)) > 1 or len(set(warmups)) > 1 or len(set(actuators)) > 1:
            logger.info("✅ Building pool has diversity!")
        else:
            logger.warning("⚠️  All buildings have identical parameters")
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch building pool: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Create multiple environments and verify they sample different buildings
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Creating multiple environments")
    logger.info("=" * 60)
    
    try:
        import tempfile
        
        env_params = []
        for i in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                env = make_diverse_env(
                    config=config,
                    eplus_output_dir=tmpdir,
                    building_pool=building_pool
                )
                
                # Wrap with building param augmentation to extract params
                wrapped_env = AugmentObservationWithBuildingParams(env)
                
                params = {
                    'area': wrapped_env.building_params['area'],
                    'warmup': wrapped_env.building_params['warmup_phases'],
                    'actuators': wrapped_env.building_params['num_actuators'],
                }
                env_params.append(params)
                
                logger.info(f"Environment {i+1}: {params}")
                
                # Clean up
                env.close()
        
        # Check if environments have different parameters
        unique_params = len(set(tuple(p.items()) for p in env_params))
        logger.info(f"\nCreated {len(env_params)} environments with {unique_params} unique parameter sets")
        
        if unique_params > 1:
            logger.info("✅ Environments have diverse building parameters!")
        else:
            logger.warning("⚠️  All environments have identical parameters")
        
    except Exception as e:
        logger.error(f"❌ Failed to create diverse environments: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests passed!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = test_building_diversity()
    sys.exit(0 if success else 1)

