import pytest
import os
import json
import shutil
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from hydra.test_utils.test_utils import TSweepRunner
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from scripts.run_baseline import main

@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def run_baseline(cfg: DictConfig) -> None:
    """Run baseline evaluation with Hydra config"""
    return main(cfg)

def test_run_baseline():
    # Load building characteristics
    with open("tests/fixtures/processed_buildings/1003000523385.json", "r") as f:
        building_characteristics = json.load(f)
    
    # Create a test config
    config = {
        "state": "VT",
        "county": "Orleans",
        "building_id": "1003000523385",
        "weather_validation": "weather_vt_1.epw",
        "reward_type": "base",
        "energy_weight": 1.0,
        "seed": 42,
        "constant": {
            "heating_setpoint": 21.0,
            "cooling_setpoint": 24.0
        },
        "hydra": {
            "run": {
                "dir": "test_baseline_results"
            }
        }
    }
    
    # Initialize and compose the config
    GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path="../configs")
    cfg = hydra.compose(config_name="baseline", overrides=[
        f"hydra.run.dir=test_baseline_results",
        f"state={config['state']}",
        f"county={config['county']}",
        f"building_id={config['building_id']}",
        f"weather_validation={config['weather_validation']}",
        f"reward_type={config['reward_type']}",
        f"energy_weight={config['energy_weight']}",
        f"seed={config['seed']}",
        f"constant.heating_setpoint={config['constant']['heating_setpoint']}",
        f"constant.cooling_setpoint={config['constant']['cooling_setpoint']}"
    ])
    
    # Run baseline evaluation
    results = main(cfg)
    
    # Load expected results from fixture
    fixture_path = "tests/fixtures/results/1003000523385_results.json"
    if not os.path.exists(fixture_path):
        # Create the fixture if it doesn't exist
        os.makedirs(os.path.dirname(fixture_path), exist_ok=True)
        with open(fixture_path, "w") as f:
            json.dump(results, f, indent=2)
        pytest.skip("Created baseline fixture. Re-run test to compare.")
    
    with open(fixture_path, "r") as f:
        expected_results = json.load(f)
    
    # Test 1: Check that results contain expected metrics
    assert isinstance(results, dict)
    expected_keys = {
        "total_reward", "mean_reward", "std_reward",
        "min_reward", "max_reward", "total_timesteps",
        "heating_setpoint", "cooling_setpoint"
    }
    assert all(key in results for key in expected_keys)
    
    # Test 2: Verify setpoints match input
    assert results["heating_setpoint"] == 21.0
    assert results["cooling_setpoint"] == 24.0
    
    # Test 3: Verify simulation ran for expected duration
    # Assuming hourly timesteps for a year
    assert results["total_timesteps"] == 8760
    
    # Test 4: Compare with fixture results
    # Allow for small numerical differences due to floating point arithmetic
    rtol = 1e-5  # relative tolerance
    atol = 1e-8  # absolute tolerance
    
    for key in ["total_reward", "mean_reward", "std_reward", "min_reward", "max_reward"]:
        np.testing.assert_allclose(
            results[key],
            expected_results[key],
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch in {key}"
        )
    
    # Exact matches for non-float values
    assert results["total_timesteps"] == expected_results["total_timesteps"]
    assert results["heating_setpoint"] == expected_results["heating_setpoint"]
    assert results["cooling_setpoint"] == expected_results["cooling_setpoint"]