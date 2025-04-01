import unittest
import pandas as pd
import os
from src.generator.search_idf import search_metadata

class TestSearchMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the fixture data
        fixture_path = os.path.join('tests', 'fixtures', 'AK.csv')
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
        self.assertEqual(results[0], "9021005053130")
        # Add specific ID assertion if you know the expected ID

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
            area=5000,
            num_floors=2,
            height=8,
            n_buildings=n_buildings
        )
        self.assertEqual(len(results), n_buildings)
        self.assertEqual(len(set(results)), n_buildings)  # Check for unique results
