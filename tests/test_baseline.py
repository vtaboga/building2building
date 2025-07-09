import unittest
import json
import shutil
from pathlib import Path
from building2building.algorithms.online.baselines import run_constant_baseline

class TestConstantBaseline(unittest.TestCase):
    def setUp(self):
        # Create temporary output directory
        self.output_dir = Path("tests/temp_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        # Clean up the temporary directory after the test
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_run_constant_baseline_basic(self):
        # Setup test paths
        building_id = "1003000512784"
        building_path = f"tests/fixtures/processed_buildings/{building_id}.epJSON"
        weather_path = "tests/fixtures/weather/weather_vt_1.epw"
        characteristics_path = f"tests/fixtures/processed_buildings/{building_id}.json"
        fixture_trajectory = f"tests/fixtures/results/{building_id}_trajectory.json"
        
        # Load building characteristics
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
        
        eplus_output_dir = self.output_dir / "eplus_output"
        
        # Run baseline
        try:
            results = run_constant_baseline(
                env_id="EnergyPlus-v0",
                path_to_building=building_path,
                path_to_weather=weather_path,
                building_characteristics=building_characteristics,
                heating_setpoint=21.0,  
                cooling_setpoint=24.0, 
                reward_type="base",
                energy_weight=1.0,
                seed=42,
                results_dir=str(self.output_dir),
                eplus_output_dir=str(eplus_output_dir)
            )
            
            # Load and compare trajectories
            with open(fixture_trajectory, 'r') as f:
                expected_trajectory = json.load(f)
            with open(self.output_dir / "trajectories/trajectories.json", 'r') as f:
                actual_trajectory = json.load(f)
            
            self.assertEqual(actual_trajectory, expected_trajectory, "Trajectory does not match expected fixture")
            
        except Exception as e:
            self.fail(f"run_constant_baseline raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
