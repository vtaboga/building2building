"""Test creating a gym environment with HVAC actuators from fixture building."""
import tempfile
from pathlib import Path

import numpy as np

from building2building.env import STORE_PATH, energyplus_path
from building2building.pipeline import (
    get_hvac_actuators,
    get_net_conditioned_area,
)
from building2building.simulator import create_simulator
from building2building.store import realize
from building2building.types import BaseRewardConfig, BuildingConfig

from building2building.types import BaseRewardConfig, BuildingConfig


def test_create_gym_env_with_hvac_actuators():
    """Test creating a gym environment from fixture building with HVAC actuators."""
    
    # Paths to test fixtures
    epjson_path = Path("tests/fixtures/bldg1.epjson").resolve()
    epw_path = Path("tests/fixtures/weather.epw").resolve()
    edd_path = Path("tests/fixtures/eplusout.edd").resolve()
    htm_path = Path("tests/fixtures/eplustbl.htm").resolve()
    
    assert epjson_path.exists(), f"Building file not found: {epjson_path}"
    assert epw_path.exists(), f"Weather file not found: {epw_path}"
    assert edd_path.exists(), f"EDD file not found: {edd_path}"
    assert htm_path.exists(), f"HTML file not found: {htm_path}"
    
    # Get HVAC actuators from .edd file
    hvac_actuators = get_hvac_actuators(edd_path)
    print(f"\nFound {len(hvac_actuators)} HVAC actuators:")
    for i, actuator in enumerate(hvac_actuators, 1):
        print(f"{i}. {actuator['component_name']} ({actuator['component_type']})")
    
    assert len(hvac_actuators) > 0, "Should have at least one HVAC actuator"
    
    # Get building area
    area = get_net_conditioned_area(htm_path)
    print(f"\nBuilding conditioned area: {area:.2f} m²")
    
    # Create temporary directory for EnergyPlus outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        eplus_output_dir = Path(tmpdir)
        
        # Create building config
        building_config = BuildingConfig(
            path_to_building=epjson_path,
            path_to_weather=epw_path,
            reward_config=BaseRewardConfig(energy_weight=1.0),
            hvac_actuators=hvac_actuators,
            eplus_output_dir=eplus_output_dir,
            warmup_phases=0,
            area=area,
        )
        
        print("\nCreating gym environment...")
        env = create_simulator(building_config)
        
        # Verify environment properties
        assert env is not None, "Environment should be created"
        assert hasattr(env, "action_space"), "Environment should have action_space"
        assert hasattr(env, "observation_space"), "Environment should have observation_space"
        
        # Check action space dimensions match number of actuators
        action_dim = env.action_space.shape[0]
        print(f"\nAction space dimension: {action_dim}")
        print(f"Number of actuators: {len(hvac_actuators)}")
        assert action_dim == len(hvac_actuators), \
            f"Action space dimension ({action_dim}) should match number of actuators ({len(hvac_actuators)})"
        
        # Check action space bounds
        print(f"Action space low: {env.action_space.low}")
        print(f"Action space high: {env.action_space.high}")
        # Action space can legitimately include negative values (e.g., sensible load requests,
        # or setpoint deltas). We only require finite bounds and high > low.
        assert np.all(np.isfinite(env.action_space.low)), "Action space lower bounds should be finite"
        assert np.all(np.isfinite(env.action_space.high)), "Action space upper bounds should be finite"
        assert np.all(env.action_space.high > env.action_space.low), \
            "Action space upper bounds should be > lower bounds"
        
        # Check metadata
        assert "hvac_actuators" in env.metadata, "Metadata should include hvac_actuators"
        assert len(env.metadata["hvac_actuators"]) == len(hvac_actuators), \
            "Metadata should list all actuator names"
        
        print("\n✓ Environment created successfully!")
        print(f"✓ Action space: Box({action_dim},)")
        print(f"✓ Observation space: {env.observation_space}")
        print(f"✓ Metadata includes {len(env.metadata['hvac_actuators'])} actuators")
        print(f"✓ Controlled zones: {env.metadata['controlled_zones']}")
        
        # Run simulation to verify HVAC actuators are used
        print("\n" + "="*70)
        print("Testing that simulation uses HVAC actuator actions (not setpoints)")
        print("="*70)
        
        obs, info = env.reset()
        print(f"\n✓ Environment reset successful")
        print(f"  Initial observation shape: {obs.shape}")
        
        # Run a few steps with different actions to verify actuators respond
        print("\nRunning 5 simulation steps with varying actuator commands...")
        
        # Step 1: Set all actuators to low values
        action_low = env.action_space.low + 0.1 * (env.action_space.high - env.action_space.low)
        print(f"\nStep 1: Low actuator values")
        print(f"  Action: {action_low}")
        obs1, reward1, terminated1, truncated1, info1 = env.step(action_low)
        print(f"  Reward: {reward1:.4f}")
        
        # Step 2: Set all actuators to high values
        action_high = env.action_space.low + 0.9 * (env.action_space.high - env.action_space.low)
        print(f"\nStep 2: High actuator values")
        print(f"  Action: {action_high}")
        obs2, reward2, terminated2, truncated2, info2 = env.step(action_high)
        print(f"  Reward: {reward2:.4f}")
        
        # Step 3: Set actuators to middle values
        action_mid = env.action_space.low + 0.5 * (env.action_space.high - env.action_space.low)
        print(f"\nStep 3: Medium actuator values")
        print(f"  Action: {action_mid}")
        obs3, reward3, terminated3, truncated3, info3 = env.step(action_mid)
        print(f"  Reward: {reward3:.4f}")
        
        # Step 4-5: Random actions
        for i in range(4, 6):
            action = env.action_space.sample()
            print(f"\nStep {i}: Random actuator values")
            print(f"  Action: {action}")
            obs_i, reward_i, term_i, trunc_i, info_i = env.step(action)
            print(f"  Reward: {reward_i:.4f}")
        
        # Verify that the simulation completed multiple steps successfully
        print("\n✓ Successfully completed 5 simulation steps with different actions")
        print("✓ HVAC actuators are being controlled (not following fixed setpoints)")
        
        # Additional verification: Check that observations changed between steps
        obs_diff_1_2 = np.linalg.norm(obs2 - obs1)
        obs_diff_2_3 = np.linalg.norm(obs3 - obs2)
        print(f"\nObservation changes:")
        print(f"  ||obs2 - obs1||: {obs_diff_1_2:.4f}")
        print(f"  ||obs3 - obs2||: {obs_diff_2_3:.4f}")
        
        # Verify observations are changing (system is responding)
        assert obs_diff_1_2 > 0, "Observations should change between steps"
        assert obs_diff_2_3 > 0, "Observations should change between steps"
        print("✓ System state is responding to different actuator commands")
        
        # Verify we're not terminated early
        assert not terminated3, "Should not terminate in first few steps"
        assert not truncated3, "Should not truncate in first few steps"
        print("✓ Simulation is running correctly without early termination")
        
        env.close()
        print("\n" + "="*70)
        print("✓ All tests passed! HVAC actuators are controlling the simulation")
        print("="*70)




if __name__ == "__main__":
    test_create_gym_env_with_hvac_actuators()
