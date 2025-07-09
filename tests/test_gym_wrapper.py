import pytest
import numpy as np
import os
import shutil
import gymnasium as gym
import json
import building2building.simulator
import rdflib
from pathlib import Path
from building2building.simulator.action_spaces import get_controllable_setpoints_rdf
from building2building.simulator import query_info 
import unittest


class TestGymWrapper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Define paths
        self.fixtures_dir = Path("tests/fixtures/processed_buildings")
        self.weather_file = Path("tests/fixtures/weather/weather_vt_1.epw")
        self.building_1z_json = self.fixtures_dir / "1003000523385.json"
        self.building_1z_epjson = self.fixtures_dir / "1003000523385.epJSON"
        self.building_2z_epjson = self.fixtures_dir / "1003000529058.epJSON"
        
        # Create temporary output directory
        self.output_dir = Path("tests/temp_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load building characteristics
        with open(self.building_1z_json, "r") as f:
            self.building_characteristics = json.load(f)
            
        # Convert buildings to RDF
        self.rdf_1z = query_info.rdf_from_json(self.building_1z_epjson)
        self.rdf_2z = query_info.rdf_from_json(self.building_2z_epjson)
        
        # Create environment
        self.env = gym.make(
            'EnergyPlus-v0',
            path_to_building=str(self.building_1z_epjson),
            path_to_weather=str(self.weather_file),
            building_characteristics=self.building_characteristics,
            reward_type="base",
            energy_weight=1.0,
        )

    def tearDown(self):
        """Clean up after tests."""
        # Close environment
        if hasattr(self, 'env'):
            self.env.close()
            
        # Clean up temporary directories
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        if Path("eplus_output").exists():
            shutil.rmtree("eplus_output")

    def test_gym_environment(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(self.env.action_space, gym.spaces.Box)
        
        for _ in range(5):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            
            if terminated or truncated:
                obs, info = self.env.reset()

    def test_get_controllable_setpoints(self):
        # Get controllable setpoints
        setpoints_1z = get_controllable_setpoints_rdf(self.rdf_1z)
        setpoints_2z = get_controllable_setpoints_rdf(self.rdf_2z)
        
        # Verify first building (1z) setpoints
        self.assertEqual(len(setpoints_1z), 1, "Expected exactly one zone in first building")
        self.assertIn("Space 0 ZN", setpoints_1z, "Expected zone 'Space 0 ZN' in first building")
        
        zone_setpoints_1z = setpoints_1z["Space 0 ZN"]
        self.assertEqual(len(zone_setpoints_1z), 2, "Expected exactly two setpoints in first building")
        
        # Check heating setpoint for first building
        heating_1z = next(sp for sp in zone_setpoints_1z if sp["setpoint_type"] == "heating")
        self.assertEqual(heating_1z["schedule_name"], "Space Type 1 Thermostat 1 Heating Setpoint")
        self.assertEqual(heating_1z["control_type"], "DualSetpoint")
        self.assertEqual(heating_1z["actuator_key"], "Zone Temperature Control,Temperature Heating Setpoint,Space 0 ZN")
        
        # Check cooling setpoint for first building
        cooling_1z = next(sp for sp in zone_setpoints_1z if sp["setpoint_type"] == "cooling")
        self.assertEqual(cooling_1z["schedule_name"], "Space Type 1 Thermostat 1 Cooling Setpoint")
        self.assertEqual(cooling_1z["control_type"], "DualSetpoint")
        self.assertEqual(cooling_1z["actuator_key"], "Zone Temperature Control,Temperature Cooling Setpoint,Space 0 ZN")

        # Verify second building (2z) setpoints
        self.assertEqual(len(setpoints_2z), 1, "Expected exactly one zone in second building")
        self.assertIn("Space 4 ZN", setpoints_2z, "Expected zone 'Space 4 ZN' in second building")
        
        zone_setpoints_2z = setpoints_2z["Space 4 ZN"]
        self.assertEqual(len(zone_setpoints_2z), 2, "Expected exactly two setpoints in second building")
        
        # Check heating setpoint for second building
        heating_2z = next(sp for sp in zone_setpoints_2z if sp["setpoint_type"] == "heating")
        self.assertEqual(heating_2z["schedule_name"], "Space Type 1 Thermostat 5 Heating Setpoint")
        self.assertEqual(heating_2z["control_type"], "DualSetpoint")
        self.assertEqual(heating_2z["actuator_key"], "Zone Temperature Control,Temperature Heating Setpoint,Space 4 ZN")
        
        # Check cooling setpoint for second building
        cooling_2z = next(sp for sp in zone_setpoints_2z if sp["setpoint_type"] == "cooling")
        self.assertEqual(cooling_2z["schedule_name"], "Space Type 1 Thermostat 5 Cooling Setpoint")
        self.assertEqual(cooling_2z["control_type"], "DualSetpoint")
        self.assertEqual(cooling_2z["actuator_key"], "Zone Temperature Control,Temperature Cooling Setpoint,Space 4 ZN")


    
