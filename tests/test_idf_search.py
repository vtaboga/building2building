import unittest
from unittest.mock import patch, Mock
import pandas as pd
import os
import pathlib
import json
from deepdiff import DeepDiff
from building2building.generator.search_idf import search_metadata, load_metadata, search_idf
from building2building.generator.processing import process_idf
from pathlib import Path
import shutil
from contextlib import contextmanager
import hashlib

def get_file_hash(filepath):
    """Get MD5 hash of a file"""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read the file in chunks
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

@contextmanager
def cd(newdir):
    """
    Context manager for changing the current working directory
    """
    prevdir = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)

class TestSearchMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the fixture data
        cls.fixtures_dir = pathlib.Path('tests/fixtures')
        cls.metadata_path = cls.fixtures_dir / 'metadata' / 'test_metadata.csv'
        cls.metadata = pd.read_csv(cls.metadata_path)

    def test_exact_match(self):
        """Test search with exact matching parameters for SmallOffice"""
        results = search_metadata(
            self.metadata,
            building_type="SmallOffice",
            area=1574.74,
            num_floors=1,
            height=10.76,
            n_buildings=1
        )
        self.assertEqual(len(results), 1)
        # Check that result is a tuple with ID and characteristics
        building_id, characteristics = results[0]
        self.assertEqual(building_id, 1002000365954)
        self.assertEqual(characteristics['building_type'], "SmallOffice")
        self.assertEqual(characteristics['num_floors'], 1)
        self.assertEqual(characteristics['area'], 1574.74)
        self.assertEqual(characteristics['height'], 10.76)

    def test_exact_match_medium_office(self):
        """Test search with exact matching parameters for MediumOffice"""
        results = search_metadata(
            self.metadata,
            building_type="MediumOffice",
            area=2676.54,
            num_floors=4,
            height=47.83,
            n_buildings=1
        )
        self.assertEqual(len(results), 1)
        building_id, characteristics = results[0]
        self.assertEqual(building_id, 1002000365947)
        self.assertEqual(characteristics['building_type'], "MediumOffice")
        self.assertEqual(characteristics['num_floors'], 4)
        self.assertEqual(characteristics['area'], 2676.54)
        self.assertEqual(characteristics['height'], 47.83)

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

    def test_multiple_buildings_same_type(self):
        """Test search returning multiple buildings of the same type using config parameters"""
        results = search_metadata(
            self.metadata,
            building_type="SmallOffice",
            area=1000,
            num_floors=1,
            height=10.0,
            n_buildings=2
        )
        self.assertEqual(len(results), 2)
        
        # Check that results are tuples with ID and characteristics
        for building_id, characteristics in results:
            self.assertIsInstance(building_id, (int, type(building_id)))
            self.assertEqual(characteristics['building_type'], "SmallOffice")
            self.assertEqual(characteristics['num_floors'], 1)
            # Area and height should be close to target values but not exact
            self.assertLess(abs(characteristics['area'] - 1000), 1000)  # Within 1000 sq ft
            self.assertLess(abs(characteristics['height'] - 10.0), 5)   # Within 5 ft

        # Get the IDs in sorted order
        found_ids = sorted(building_id for building_id, _ in results)
        # These should be the IDs of single zone files in fixtures
        expected_ids = [1002000365954, 1003000523385]  # IDs from fixtures
        self.assertEqual(found_ids, expected_ids)

    def test_search_fewer_than_requested(self):
        """Test search requesting more buildings than available of a specific type"""
        results = search_metadata(
            self.metadata,
            building_type="Hospital",
            area=3000.0,
            num_floors=2,
            height=40.0,
            n_buildings=2  # Request 2 but expect only 1
        )
        self.assertEqual(len(results), 1)  # Should only find one hospital
        
        # Check that result is a tuple with ID and characteristics
        building_id, characteristics = results[0]
        self.assertEqual(building_id, 1002000365956)  # ID of the only hospital in test metadata
        self.assertEqual(characteristics['building_type'], "Hospital")
        self.assertEqual(characteristics['num_floors'], 2)
        self.assertEqual(characteristics['area'], 3000.00)
        self.assertEqual(characteristics['height'], 40.00)



class TestMetadataLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = pathlib.Path('tests/fixtures')
        cls.metadata_path = cls.fixtures_dir / 'metadata' / 'test_metadata.csv'

    def test_load_metadata_all_counties(self):
        """Test loading metadata without county filtering"""
        df = load_metadata(self.metadata_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(df['State_Abbr'].unique()[0], 'VT')
        # Verify county column exists and has values
        self.assertIn('County', df.columns)
        self.assertTrue(all(df['County'].notna()))

    def test_load_metadata_with_county(self):
        """Test loading metadata with county filter"""
        df = load_metadata(self.metadata_path, county='Caledonia')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertTrue(all(df['County'] == 'Caledonia'))

    def test_load_metadata_nonexistent_county(self):
        """Test loading metadata with a county that doesn't exist in the data"""
        df = load_metadata(self.metadata_path, county='Orleans')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)  # Should return empty DataFrame for non-existent county

class TestSearchIDF(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = pathlib.Path('tests/fixtures')
        self.metadata_path = self.fixtures_dir / 'test_metadata.csv'
        # Define fixture paths for test files
        self.fixture_idf_1z = self.fixtures_dir / 'idf' / 'building_1z.idf'
        self.fixture_epjson_1z = self.fixtures_dir / 'processed_buildings' / 'building_1z.epJSON'
        self.fixture_idf_5z = self.fixtures_dir / 'idf' / 'building_5z.idf'
        self.fixture_epjson_5z = self.fixtures_dir / 'processed_buildings' / 'building_5z.epJSON'
        self.fixture_weather = self.fixtures_dir / 'weather_vt.epw'

    @patch('building2building.generator.search_idf.download_and_extract_county_idf')
    @patch('building2building.generator.search_idf.download_metadata')
    @patch('building2building.generator.search_idf.process_metadata')
    @patch('building2building.generator.search_idf.process_idf')
    @patch('building2building.generator.search_idf.download_epw')
    @patch('os.path.join')
    def test_search_idf_small_office(self, mock_path_join, mock_download_epw, mock_process_idf, 
                            mock_process_metadata, mock_download_metadata, 
                            mock_download_county):
        """Test search_idf with SmallOffice parameters"""
        # Set up mocks
        mock_download_county.return_value = self.fixtures_dir
        mock_process_metadata.return_value = self.metadata_path
        mock_process_idf.return_value = [str(self.fixture_epjson_1z)]
        mock_download_epw.return_value = [str(self.fixture_weather)]
        mock_path_join.return_value = str(self.fixture_epjson_1z)

        buildings, weather_files = search_idf(
            state='VT',
            county='Caledonia',
            building_type='SmallOffice',
            area=1574.74,
            num_floors=1,
            height=10.76,
            n_buildings=1,
            n_weather_files=1,
            keep_original=True
        )
        
        # Check return structure
        self.assertIsInstance(buildings, list)
        self.assertIsInstance(weather_files, list)
        
        # Check building data structure
        if buildings:
            building_path, characteristics = buildings[0]
            self.assertEqual(building_path, str(self.fixture_epjson_1z))
            self.assertEqual(characteristics['building_type'], 'SmallOffice')
            self.assertEqual(characteristics['num_floors'], 1)
            self.assertEqual(characteristics['area'], 1574.74)
            self.assertEqual(characteristics['height'], 10.76)
            
        # Check weather files
        if weather_files:
            self.assertEqual(weather_files[0], str(self.fixture_weather))

        # Verify mocks were called correctly
        mock_download_county.assert_called_once_with('VT', 'Caledonia')
        mock_download_metadata.assert_called_once_with(state='VT')
        mock_process_metadata.assert_called_once_with(state='VT')
        self.assertEqual(mock_download_epw.call_count, 1)

    @patch('building2building.generator.search_idf.download_and_extract_county_idf')
    @patch('building2building.generator.search_idf.download_metadata')
    @patch('building2building.generator.search_idf.process_metadata')
    @patch('building2building.generator.search_idf.process_idf')
    @patch('building2building.generator.search_idf.download_epw')
    @patch('os.path.join')
    def test_search_idf_medium_office(self, mock_path_join, mock_download_epw, mock_process_idf, 
                                        mock_process_metadata, mock_download_metadata, 
                                        mock_download_county):
        """Test search_idf with MediumOffice parameters"""
        # Set up mocks
        mock_download_county.return_value = self.fixtures_dir
        mock_process_metadata.return_value = self.metadata_path
        mock_process_idf.return_value = [str(self.fixture_epjson_5z)]
        mock_download_epw.return_value = [str(self.fixture_weather)]
        mock_path_join.return_value = str(self.fixture_epjson_5z)

        buildings, weather_files = search_idf(
            state='VT',
            county='Caledonia',
            building_type='MediumOffice',
            area=2676.54,
            num_floors=4,
            height=47.83,
            n_buildings=1,
            n_weather_files=1,
            keep_original=True
        )
        
        # Check return structure
        self.assertIsInstance(buildings, list)
        self.assertIsInstance(weather_files, list)
        
        # Check building data structure
        if buildings:
            building_path, characteristics = buildings[0]
            self.assertEqual(building_path, str(self.fixture_epjson_5z))
            self.assertEqual(characteristics['building_type'], 'MediumOffice')
            self.assertEqual(characteristics['num_floors'], 4)
            self.assertEqual(characteristics['area'], 2676.54)
            self.assertEqual(characteristics['height'], 47.83)
            
        # Check weather files
        if weather_files:
            self.assertEqual(weather_files[0], str(self.fixture_weather))

        # Verify mocks were called correctly
        mock_download_county.assert_called_once_with('VT', 'Caledonia')
        mock_download_metadata.assert_called_once_with(state='VT')
        mock_process_metadata.assert_called_once_with(state='VT')
        self.assertEqual(mock_download_epw.call_count, 1)

    @patch('building2building.generator.search_idf.download_and_extract_county_idf')
    @patch('building2building.generator.search_idf.download_metadata')
    def test_search_idf_invalid_params(self, mock_download_metadata, mock_download_county):
        """Test search with invalid parameters"""
        mock_download_county.side_effect = Exception("Invalid state or county")
        
        with self.assertRaises(Exception):
            search_idf(
                state='INVALID',
                county='INVALID',
                building_type='NonExistentType',
                area=0,
                num_floors=0,
                height=None,
                n_buildings=1,
                n_weather_files=1,
                keep_original=True
            )

    @patch('building2building.generator.search_idf.download_and_extract_county_idf')
    @patch('building2building.generator.search_idf.download_metadata')
    @patch('building2building.generator.search_idf.process_metadata')
    def test_search_idf_nonexistent_county(self, mock_process_metadata, mock_download_metadata, mock_download_county):
        """Test search_idf with a county that doesn't exist in the metadata"""
        mock_download_county.return_value = self.fixtures_dir
        mock_process_metadata.return_value = self.metadata_path

        with self.assertRaises(Exception) as context:
            search_idf(
                state='VT',
                county='Orleans',  # County not in test metadata
                building_type='SmallOffice',
                area=1000,
                num_floors=1,
                height=10.0,
                n_buildings=1,
                n_weather_files=1,
                keep_original=True
            )
        
        # Verify the error message indicates no buildings were found
        self.assertIn("No matching IDF files found", str(context.exception))

class TestProcessIDF(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.fixtures_dir = Path('tests/fixtures').resolve()  # Get absolute path
        # Create main output directory
        self.test_output_dir = self.fixtures_dir / 'test_output' / 'VT' / 'TestCounty'
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        # Create temp directory for transitions
        self.temp_dir = self.test_output_dir / 'temp'
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test files"""
        # Remove the test_output directory
        test_output_root = self.fixtures_dir / 'test_output'
        if test_output_root.exists():
            shutil.rmtree(test_output_root)
            
        # Remove temp_transition directory in fixtures/idf if it exists
        temp_transition_dir = self.fixtures_dir / 'idf' / 'temp_transition'
        if temp_transition_dir.exists():
            shutil.rmtree(temp_transition_dir)

    def test_process_single_idf(self):
        """Test processing a single IDF file"""
        # Test data
        building_id = 1003000523385
        test_characteristics = {
            "building_type": "SmallOffice",
            "num_floors": 1,
            "area": 1000.0,
            "height": 10.0
        }
        
        # Verify input file exists
        input_dir = self.fixtures_dir / 'idf'
        input_file = input_dir / f"{building_id}.idf"
        print(f"\nDebug - Input file path: {input_file}")
        print(f"Debug - Input file exists: {input_file.exists()}")
        print(f"Debug - Current working dir: {os.getcwd()}")
        
        # Process the IDF file
        with cd(self.temp_dir):  # Change to temp directory for processing
            processed_paths = process_idf(
                [(building_id, test_characteristics)],
                state="VT",
                county="TestCounty",
                keep_original=False,
                output_dir=self.test_output_dir,
                input_dir=input_dir
            )
            print(f"Debug - Working dir during processing: {os.getcwd()}")
        
        # Check that we got a result
        self.assertEqual(len(processed_paths), 1)
        
        # Check that output file exists and is epJSON
        output_path = processed_paths[0]
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.suffix, '.epJSON')
        
        # Check that characteristics file was created
        characteristics_path = self.test_output_dir / f"{building_id}.json"
        self.assertTrue(characteristics_path.exists())

        # Compare with reference files
        ref_dir = self.fixtures_dir / 'processed_buildings'
        ref_epjson = ref_dir / f"{building_id}.epJSON"

        # Compare epJSON files by loading and comparing contents
        with open(output_path) as f1, open(ref_epjson) as f2:
            test_epjson = json.load(f1)
            ref_epjson_content = json.load(f2)
            diff = DeepDiff(test_epjson, ref_epjson_content, significant_digits=6)
            self.assertEqual(diff, {}, "Generated epJSON file differs from reference")

       
