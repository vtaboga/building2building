import unittest
import pandas as pd
import os
from building2building.generator.search_idf import search_metadata

class TestSearchMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the fixture data
        fixture_path = os.path.join('tests', 'fixtures', 'test_metadata.csv')
        cls.metadata = pd.read_csv(fixture_path)

    def test_exact_match(self):
        """Test search with exact matching parameters"""
        results = search_metadata(
            self.metadata,
            building_type="SmallHotel",
            area=1140,
            num_floors=6,
            n_buildings=1
        )
        self.assertEqual(len(results), 1)
        # Check that result is a tuple with ID and characteristics
        building_id, characteristics = results[0]
        self.assertEqual(building_id, 9021005053133)
        self.assertEqual(characteristics['building_type'], "SmallHotel")
        self.assertEqual(characteristics['num_floors'], 6)

    def test_no_matches(self):
        """Test search with non-existent building type"""
        results = search_metadata(
            self.metadata,
            building_type="NonExistentType",
            area=5000,
            num_floors=2,
            height=8,
            n_buildings=1
        )
        self.assertEqual(len(results), 0)

    def test_multiple_results(self):
        """Test search returning multiple buildings"""
        n_buildings = 3
        results = search_metadata(
            self.metadata,
            building_type="MediumOffice",
            area=500,
            num_floors=4,
            height=50,
            n_buildings=n_buildings
        )
        self.assertEqual(len(results), n_buildings)
        # Check all results are tuples with correct structure
        for building_id, characteristics in results:
            self.assertIsInstance(building_id, (int, type(building_id))) 
            self.assertEqual(characteristics['building_type'], "MediumOffice")
            self.assertIn('num_floors', characteristics)
            self.assertIn('area', characteristics)
            self.assertIn('height', characteristics)
        # Check for unique results
        building_ids = [result[0] for result in results]
        self.assertEqual(len(set(building_ids)), n_buildings)
