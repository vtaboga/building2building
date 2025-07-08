import pytest
import os
import json
import shutil
from pathlib import Path
from building2building.algorithms.online.baselines import run_constant_baseline

def test_run_baseline():
    # Load building characteristics
    with open("tests/fixtures/building_1z.json", "r") as f:
        building_characteristics = json.load(f)
    
    # Setup test directories
    test_results_dir = "test_baseline_results"
    test_eplus_output = os.path.join(test_results_dir, "eplus_output")
    
    try:
        # Run baseline evaluation
        results = run_constant_baseline(
            env_id="EnergyPlus-v0",
            path_to_building="tests/fixtures/building_1z.epJSON",
            path_to_weather="tests/fixtures/alaska.epw",
            building_characteristics=building_characteristics,
            heating_setpoint=21.0,
            cooling_setpoint=24.0,
            reward_type="base",
            energy_weight=1.0,
            seed=42,
            results_dir=test_results_dir,
            eplus_output_dir=test_eplus_output
        )
        
        # Test 1: Check that results contain expected metrics
        assert isinstance(results, dict)
        expected_keys = {
            "total_reward", "mean_reward", "std_reward",
            "min_reward", "max_reward", "total_timesteps",
            "heating_setpoint", "cooling_setpoint"
        }
        assert all(key in results for key in expected_keys)
        
        # Test 2: Verify output directory structure
        assert os.path.exists(test_results_dir)
        assert os.path.exists(test_eplus_output)
        
        # Test 3: Check results file exists
        results_file = os.path.join(test_results_dir, "baseline_results.json")
        assert os.path.exists(results_file)
        
        # Test 4: Verify trajectories were created
        trajectories_dir = os.path.join(test_results_dir, "trajectories")
        assert os.path.exists(trajectories_dir)
        assert any(f.endswith('.json') for f in os.listdir(trajectories_dir))
        
        # Test 5: Verify setpoints match input
        assert results["heating_setpoint"] == 21.0
        assert results["cooling_setpoint"] == 24.0
        
        # Test 6: Verify simulation ran for expected duration
        # Assuming hourly timesteps for a year
        assert results["total_timesteps"] == 8760
        
    finally:
        # Cleanup test directories
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir) 