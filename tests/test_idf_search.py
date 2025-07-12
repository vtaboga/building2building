# import hashlib
# import json
# import os
# import pathlib
# import shutil
# import unittest
# from contextlib import contextmanager
# from pathlib import Path
# from unittest.mock import Mock, patch

# import pandas as pd
# import pytest
# from building2building.generator.search_idf import (
#     process_idf,
#     search_idf,
#     search_metadata,
# )
# from deepdiff import DeepDiff


# def get_file_hash(filepath):
#     """Get MD5 hash of a file"""
#     md5_hash = hashlib.md5()
#     with open(filepath, "rb") as f:
#         # Read the file in chunks
#         for chunk in iter(lambda: f.read(4096), b""):
#             md5_hash.update(chunk)
#     return md5_hash.hexdigest()


# @contextmanager
# def cd(newdir):
#     """
#     Context manager for changing the current working directory
#     """
#     prevdir = os.getcwd()
#     os.chdir(newdir)
#     try:
#         yield
#     finally:
#         os.chdir(prevdir)


# @pytest.fixture
# def fixtures_dir():
#     return Path("tests/fixtures")


# @pytest.fixture
# def metadata_path(fixtures_dir):
#     path = fixtures_dir / "metadata" / "test_metadata.csv"
#     return path


# def test_exact_match(metadata_path):
#     """Test search with exact matching parameters for SmallOffice"""

#     results = search_metadata(
#         metadata_path,
#         building_type="SmallOffice",
#         area=1574.74,
#         num_floors=1,
#         height=10.76,
#         n_buildings=1,
#     )
#     assert len(results) == 1
#     # Check that result is a tuple with ID and characteristics
#     building_id, characteristics = results[0]
#     assert building_id == 1002000365954
#     assert characteristics.building_type == "SmallOffice"
#     assert characteristics.num_floors == 1
#     assert characteristics.area == 1574.74
#     assert characteristics.height == 10.76


# def test_exact_match_medium_office(metadata_path):
#     """Test search with exact matching parameters for MediumOffice"""
#     results = search_metadata(
#         metadata_path,
#         building_type="MediumOffice",
#         area=2676.54,
#         num_floors=4,
#         height=47.83,
#         n_buildings=1,
#     )
#     assert len(results) == 1
#     building_id, characteristics = results[0]
#     assert building_id == 1002000365947
#     assert characteristics.building_type == "MediumOffice"
#     assert characteristics.num_floors == 4
#     assert characteristics.area == 2676.54
#     assert characteristics.height == 47.83


# def test_no_matches(metadata_path):
#     """Test search with non-existent building type"""
#     results = search_metadata(
#         metadata_path,
#         building_type="NonExistentType",
#         area=5000,
#         num_floors=2,
#         height=8,
#         n_buildings=1,
#     )
#     assert len(results) == 0


# def test_multiple_buildings_same_type(metadata_path):
#     """Test search returning multiple buildings of the same type using config parameters"""
#     results = search_metadata(
#         metadata_path,
#         building_type="SmallOffice",
#         area=1000,
#         num_floors=1,
#         height=10.0,
#         n_buildings=2,
#     )
#     assert len(results) == 2

#     # Check that results are tuples with ID and characteristics
#     for building_id, characteristics in results:
#         assert isinstance(building_id, int)
#         assert characteristics.building_type == "SmallOffice"
#         assert characteristics.num_floors == 1
#         # Area and height should be close to target values but not exact
#         assert abs(characteristics.area - 1000) < 1000  # Within 1000 sq ft
#         assert abs(characteristics.height - 10.0) < 5  # Within 5 ft

#     # Get the IDs in sorted order
#     found_ids = sorted(
#         [int(building_id) for building_id, _ in results]
#     )  # Convert to int for proper comparison
#     # These should be the IDs of single zone files in fixtures
#     expected_ids = [1002000365954, 1003000523385]  # IDs from fixtures
#     assert found_ids == expected_ids


# def test_search_fewer_than_requested(metadata_path):
#     """Test search requesting more buildings than available of a specific type"""
#     results = search_metadata(
#         metadata_path,
#         building_type="Hospital",
#         area=3000.0,
#         num_floors=2,
#         height=40.0,
#         n_buildings=2,  # Request 2 but expect only 1
#     )
#     assert len(results) == 1  # Should only find one hospital

#     # Check that result is a tuple with ID and characteristics
#     building_id, characteristics = results[0]
#     assert building_id == 1002000365956  # ID of the only hospital in test metadata
#     assert characteristics.building_type == "Hospital"
#     assert characteristics.num_floors == 2
#     assert characteristics.area == 3000.00
#     assert characteristics.height == 40.00


# def test_sorting_order(metadata_path):
#     """Test that buildings are returned in correct order based on similarity to criteria"""
#     results = search_metadata(
#         metadata_path,
#         building_type="SmallOffice",
#         area=1500,  # Target values that don't exactly match any building
#         num_floors=1,
#         height=11.0,
#         n_buildings=3,
#     )

#     assert len(results) >= 2, "Should find at least 2 buildings for comparison"

#     # Get the differences for each result
#     diffs = []
#     for building_id, chars in results:
#         area_diff = abs(chars.area - 1500)
#         height_diff = abs(chars.height - 11.0)
#         diffs.append((area_diff, height_diff))

#     # Check that results are sorted by area difference first, then height
#     for i in range(len(diffs) - 1):
#         # Either the first building should have smaller area difference
#         # or equal area difference but smaller height difference
#         assert diffs[i][0] < diffs[i + 1][0] or (
#             diffs[i][0] == diffs[i + 1][0] and diffs[i][1] <= diffs[i + 1][1]
#         ), f"Results not properly sorted at position {i}"


# def test_edge_cases(metadata_path):
#     """Test search_metadata with edge cases and extreme values"""
#     # Test with zero values
#     zero_results = search_metadata(
#         metadata_path,
#         building_type="SmallOffice",
#         area=0,
#         num_floors=0,
#         height=0,
#         n_buildings=1,
#     )
#     assert len(zero_results) >= 0

#     # Test with negative values
#     negative_results = search_metadata(
#         metadata_path,
#         building_type="SmallOffice",
#         area=-1000,
#         num_floors=-1,
#         height=-10,
#         n_buildings=1,
#     )
#     assert len(negative_results) >= 0

#     # Test with None for optional parameters
#     none_results = search_metadata(
#         metadata_path,
#         building_type="SmallOffice",
#         area=None,
#         num_floors=None,
#         height=None,
#         n_buildings=1,
#     )
#     assert len(none_results) >= 0

#     # Test with extreme values
#     extreme_results = search_metadata(
#         metadata_path,
#         building_type="SmallOffice",
#         area=1e6,  # Very large area
#         num_floors=100,  # Very tall building
#         height=1000,  # Extreme height
#         n_buildings=1,
#     )
#     assert len(extreme_results) >= 0

#     # For any results returned, verify the structure is correct
#     for results in [zero_results, negative_results, none_results, extreme_results]:
#         if results:
#             building_id, characteristics = results[0]
#             assert isinstance(building_id, int)
#             assert characteristics.building_type == "SmallOffice"


# def test_load_metadata_all_counties(metadata_path):
#     """Test loading metadata without county filtering"""
#     df = pd.read_csv(metadata_path)
#     assert not df.empty
#     assert df["State_Abbr"].unique()[0] == "VT"
#     # Verify county column exists and has values
#     assert "County" in df.columns
#     assert all(df["County"].notna())


# def test_load_metadata_nonexistent_county(metadata_path):
#     """Test loading metadata with a county that doesn't exist in the data"""
#     list = search_metadata(metadata_path, county="Does not exist")
#     assert len(list) == 0


# @pytest.fixture
# def search_fixtures(fixtures_dir, metadata_path):
#     return {
#         "idf_1z": fixtures_dir / "idf" / "1003000523385.idf",
#         "epjson_1z": fixtures_dir / "processed_buildings" / "1003000523385.epJSON",
#         "idf_5z": fixtures_dir / "idf" / "1003000523385.idf",
#         "epjson_5z": fixtures_dir / "processed_buildings" / "1003000523385.epJSON",
#         "weather_1": fixtures_dir / "weather" / "weather_vt_1.epw",
#         "weather_2": fixtures_dir / "weather" / "weather_vt_2.epw",
#     }


# @pytest.fixture
# def search_idf_mock(mocker):
#     return {
#         "download_county": mocker.patch(
#             "building2building.generator.search_idf.download_and_extract_county_idf"
#         ),
#         "download_metadata": mocker.patch(
#             "building2building.generator.search_idf.download_metadata"
#         ),
#         "process_metadata": mocker.patch(
#             "building2building.generator.processing.process_metadata"
#         ),
#         "process_idf": mocker.patch(
#             "building2building.generator.search_idf.process_idf"
#         ),
#         "download_epw": mocker.patch(
#             "building2building.generator.search_idf.download_epw"
#         ),
#     }


# def test_search_idf_small_office(
#     search_idf_mock,
#     fixtures_dir,
#     metadata_path,
#     search_fixtures,
# ):
#     """Test search_idf with SmallOffice parameters"""
#     # Set up mocks
#     search_idf_mock["download_county"].return_value = fixtures_dir
#     search_idf_mock["process_metadata"].return_value = metadata_path
#     search_idf_mock["process_idf"].return_value = search_fixtures["epjson_1z"]
#     search_idf_mock["download_epw"].return_value = search_fixtures["weather_1"]

#     buildings = search_idf(
#         state="VT",
#         county="Caledonia",
#         building_type="SmallOffice",
#         area=1574.74,
#         num_floors=1,
#         height=10.76,
#         n_buildings=1,
#     )

#     # Check return structure
#     assert isinstance(buildings, list)

#     # Check building data structure
#     if buildings:
#         building_path, characteristics = buildings[0]
#         assert building_path == search_fixtures["epjson_1z"]
#         assert characteristics.building_type == "SmallOffice"
#         assert characteristics.num_floors == 1
#         assert characteristics.area == 1574.74
#         assert characteristics.height == 10.76


# @patch("building2building.generator.search_idf.download_and_extract_county_idf")
# @patch("building2building.generator.search_idf.download_metadata")
# @patch("building2building.generator.search_idf.process_metadata")
# @patch("building2building.generator.search_idf.process_idf")
# @patch("building2building.generator.search_idf.download_epw")
# @patch("os.path.join")
# def test_search_idf_medium_office(
#     fixtures_dir,
#     metadata_path,
#     search_idf_mock,
#     search_fixtures,
# ):
#     mock_download_epw = search_idf_mock["download_epw"]
#     mock_process_idf = search_idf_mock["process_idf"]
#     mock_process_metadata = search_idf_mock["process_metadata"]
#     mock_download_metadata = search_idf_mock["download_metadata"]
#     mock_download_county = search_idf_mock["download_county"]

#     """Test search_idf with MediumOffice parameters"""
#     # Set up mocks
#     mock_download_county.return_value = fixtures_dir
#     mock_process_metadata.return_value = metadata_path
#     mock_process_idf.return_value = [str(self.fixture_epjson_5z)]
#     mock_download_epw.return_value = [str(self.fixture_weather_1)]

#     buildings = search_idf(
#         state="VT",
#         county="Caledonia",
#         building_type="MediumOffice",
#         area=2676.54,
#         num_floors=4,
#         height=47.83,
#         n_buildings=1,
#     )

#     # Check return structure
#     assert isinstance(buildings, list)

#     # Check building data structure
#     if buildings:
#         building_path, characteristics = buildings[0]
#         assert building_path == str(self.fixture_epjson_5z)
#         assert characteristics.building_type == "MediumOffice"
#         assert characteristics.num_floors == 4
#         assert characteristics.area == 2676.54
#         assert characteristics.height == 47.83


# def test_search_idf_invalid_params():
#     """Test search with invalid parameters"""
#     mock_download_county.side_effect = Exception("Invalid state or county")

#     with self.assertRaises(Exception):
#         search_idf(
#             state="INVALID",
#             county="INVALID",
#             building_type="NonExistentType",
#             area=0,
#             num_floors=0,
#             height=None,
#             n_buildings=1,
#             n_weather_files=1,
#             keep_original=True,
#         )


# @patch("building2building.generator.search_idf.download_and_extract_county_idf")
# @patch("building2building.generator.search_idf.download_metadata")
# @patch("building2building.generator.search_idf.process_metadata")
# def test_search_idf_nonexistent_county(
#     self, mock_process_metadata, mock_download_metadata, mock_download_county
# ):
#     """Test search_idf with a county that doesn't exist in the metadata"""
#     mock_download_county.return_value = self.fixtures_dir
#     mock_process_metadata.return_value = self.metadata_path

#     with self.assertRaises(Exception) as context:
#         search_idf(
#             state="VT",
#             county="Orleans",  # County not in test metadata
#             building_type="SmallOffice",
#             area=1000,
#             num_floors=1,
#             height=10.0,
#             n_buildings=1,
#             n_weather_files=1,
#             keep_original=True,
#         )

#     # Verify the error message indicates no buildings were found
#     self.assertIn("No matching IDF files found", str(context.exception))


# @patch("building2building.generator.search_idf.download_and_extract_county_idf")
# @patch("building2building.generator.search_idf.download_metadata")
# @patch("building2building.generator.search_idf.process_metadata")
# @patch("building2building.generator.search_idf.process_idf")
# def test_corrupted_idf_files(
#     self,
#     mock_process_idf,
#     mock_process_metadata,
#     mock_download_metadata,
#     mock_download_county,
# ):
#     """Test handling of corrupted and missing IDF files"""
#     # Create a temporary corrupted IDF file
#     import tempfile

#     with tempfile.NamedTemporaryFile(mode="w", suffix=".idf", delete=False) as f:
#         # Write invalid IDF content
#         f.write("This is not a valid IDF file content\n")
#         f.write("It should cause processing to fail\n")
#         corrupted_path = f.name

#     try:
#         # Set up mocks
#         mock_download_county.return_value = self.fixtures_dir
#         mock_process_metadata.return_value = self.metadata_path

#         # Mock process_idf to simulate failure
#         mock_process_idf.side_effect = Exception("Failed to process corrupted IDF file")

#         with self.assertRaises(Exception) as context:
#             search_idf(
#                 state="VT",
#                 county="Caledonia",
#                 building_type="SmallOffice",
#                 area=1500,
#                 num_floors=1,
#                 height=10.0,
#                 n_buildings=1,
#                 n_weather_files=1,
#                 keep_original=True,
#             )

#         # Verify that the error is related to IDF processing
#         self.assertIn("Failed to process", str(context.exception))

#         # Test with missing IDF file
#         mock_process_idf.reset_mock()
#         mock_process_idf.side_effect = FileNotFoundError("IDF file not found")

#         with self.assertRaises(Exception) as context:
#             search_idf(
#                 state="VT",
#                 county="Caledonia",
#                 building_type="SmallOffice",
#                 area=1500,
#                 num_floors=1,
#                 height=10.0,
#                 n_buildings=1,
#                 n_weather_files=1,
#                 keep_original=True,
#             )

#         # Verify that the error is related to missing file
#         self.assertIn("not found", str(context.exception).lower())

#     finally:
#         # Clean up the temporary file
#         os.unlink(corrupted_path)


# @patch("building2building.generator.search_idf.download_and_extract_county_idf")
# @patch("building2building.generator.search_idf.download_metadata")
# def test_invalid_state_code(self, mock_download_metadata, mock_download_county):
#     """Test search with invalid state code"""
#     mock_download_metadata.side_effect = Exception("Invalid state code")

#     with self.assertRaises(Exception) as context:
#         search_idf(
#             state="XX",  # Invalid state code
#             county="Caledonia",
#             building_type="SmallOffice",
#             area=1500,
#             num_floors=1,
#             height=10.0,
#             n_buildings=1,
#             n_weather_files=1,
#             keep_original=True,
#         )

#     self.assertIn("Invalid state code", str(context.exception))


# @patch("building2building.generator.search_idf.download_and_extract_county_idf")
# @patch("building2building.generator.search_idf.download_metadata")
# @patch("building2building.generator.search_idf.process_metadata")
# def test_invalid_building_type(
#     self, mock_process_metadata, mock_download_metadata, mock_download_county
# ):
#     """Test search with non-existent building type"""
#     mock_download_county.return_value = self.fixtures_dir
#     mock_process_metadata.return_value = self.metadata_path

#     with self.assertRaises(Exception) as context:
#         search_idf(
#             state="VT",
#             county="Caledonia",
#             building_type="NonExistentBuildingType",  # Invalid building type
#             area=1500,
#             num_floors=1,
#             height=10.0,
#             n_buildings=1,
#             n_weather_files=1,
#             keep_original=True,
#         )

#     self.assertIn("No matching IDF files found", str(context.exception))


# @patch("building2building.generator.search_idf.download_and_extract_county_idf")
# @patch("building2building.generator.search_idf.download_metadata")
# @patch("building2building.generator.search_idf.process_metadata")
# @patch("building2building.generator.search_idf.download_epw")
# def test_download_failures(
#     self,
#     mock_download_epw,
#     mock_process_metadata,
#     mock_download_metadata,
#     mock_download_county,
# ):
#     """Test handling of download failures"""
#     # Test metadata download failure
#     mock_download_metadata.side_effect = Exception("Failed to download metadata")

#     with self.assertRaises(Exception) as context:
#         search_idf(
#             state="VT",
#             county="Caledonia",
#             building_type="SmallOffice",
#             area=1500,
#             num_floors=1,
#             height=10.0,
#             n_buildings=1,
#             n_weather_files=1,
#             keep_original=True,
#         )

#     self.assertIn("Failed to download metadata", str(context.exception))

#     # Test weather file download failure
#     mock_download_metadata.reset_mock()
#     mock_download_metadata.side_effect = None
#     mock_process_metadata.return_value = self.metadata_path
#     mock_download_county.return_value = self.fixtures_dir
#     mock_download_epw.side_effect = Exception("Failed to download weather files")

#     with self.assertRaises(Exception) as context:
#         search_idf(
#             state="VT",
#             county="Caledonia",
#             building_type="SmallOffice",
#             area=1500,
#             num_floors=1,
#             height=10.0,
#             n_buildings=1,
#             n_weather_files=1,
#             keep_original=True,
#         )

#     self.assertIn("Failed to download weather files", str(context.exception))


# class TestProcessIDF(unittest.TestCase):
#     def setUp(self):
#         """Set up test fixtures"""
#         self.fixtures_dir = Path("tests/fixtures").resolve()  # Get absolute path
#         # Create main output directory
#         self.test_output_dir = self.fixtures_dir / "test_output" / "VT" / "TestCounty"
#         self.test_output_dir.mkdir(parents=True, exist_ok=True)
#         # Create temp directory for transitions
#         self.temp_dir = self.test_output_dir / "temp"
#         self.temp_dir.mkdir(exist_ok=True)

#     def tearDown(self):
#         """Clean up test files"""
#         # Remove the test_output directory
#         test_output_root = self.fixtures_dir / "test_output"
#         if test_output_root.exists():
#             shutil.rmtree(test_output_root)

#         # Remove temp_transition directory in fixtures/idf if it exists
#         temp_transition_dir = self.fixtures_dir / "idf" / "temp_transition"
#         if temp_transition_dir.exists():
#             shutil.rmtree(temp_transition_dir)

#     def test_process_single_idf(self):
#         """Test processing a single IDF file"""
#         # Test data
#         building_id = 1003000523385

#         # Load the actual characteristics from the reference JSON file
#         ref_characteristics_path = (
#             self.fixtures_dir / "processed_buildings" / f"{building_id}.json"
#         )
#         with open(ref_characteristics_path) as f:
#             test_characteristics = json.load(f)

#         # Verify input file exists
#         input_dir = self.fixtures_dir / "idf"
#         input_file = input_dir / f"{building_id}.idf"

#         # Process the IDF file
#         with cd(self.temp_dir):  # Change to temp directory for processing
#             processed_paths = process_idf(
#                 [(building_id, test_characteristics)],
#                 state="VT",
#                 county="TestCounty",
#                 keep_original=True,
#                 output_dir=self.test_output_dir,
#                 input_dir=input_dir,
#             )

#         # Check that we got a result
#         self.assertEqual(len(processed_paths), 1)

#         # Check that output file exists and is epJSON
#         output_path = processed_paths[0]
#         self.assertTrue(output_path.exists())
#         self.assertEqual(output_path.suffix, ".epJSON")

#         # Check that characteristics file was created
#         characteristics_path = self.test_output_dir / f"{building_id}.json"
#         self.assertTrue(characteristics_path.exists())

#         # Compare with reference files
#         ref_dir = self.fixtures_dir / "processed_buildings"
#         ref_epjson = ref_dir / f"{building_id}.epJSON"

#         # Compare epJSON files by loading and comparing contents
#         with open(output_path) as f1, open(ref_epjson) as f2:
#             test_epjson = json.load(f1)
#             ref_epjson_content = json.load(f2)
#             diff = DeepDiff(test_epjson, ref_epjson_content, significant_digits=6)
#             self.assertEqual(diff, {}, "Generated epJSON file differs from reference")
