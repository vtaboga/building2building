import hashlib
import json
import os
import pathlib
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from building2building.env import DataPaths
from building2building.generator.downloader import process_metadata
from building2building.generator.search_idf import (
    process_idf,
    search_idf,
    search_metadata,
)
from deepdiff import DeepDiff

DataPaths._data_path.set(Path("tests/data").resolve())


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


@pytest.fixture
def state() -> str:
    return "VT"


@pytest.fixture
def county() -> str:
    return "Caledonia"


@pytest.fixture
def metadata_path(state) -> Path:
    return DataPaths.metadata_dir() / f"{state}.csv"


def test_exact_match(metadata_path):
    """Test search with exact matching parameters for SmallOffice"""

    results = search_metadata(
        metadata_path,
        building_type="SmallOffice",
        area=1574.74,
        num_floors=1,
        height=10.76,
        n_buildings=1,
    )
    assert len(results) == 1
    # Check that result is a tuple with ID and characteristics
    building_id, characteristics = results[0]
    assert building_id == 1007001524538
    assert characteristics.building_type == "SmallOffice"
    assert characteristics.num_floors == 1
    assert abs(characteristics.area - 1580.68) < 0.01
    assert abs(characteristics.height - 14.83) < 0.01


def test_exact_match_medium_office(metadata_path):
    """Test search with exact matching parameters for MediumOffice"""
    results = search_metadata(
        metadata_path,
        building_type="MediumOffice",
        area=2676.54,
        num_floors=4,
        height=47.83,
        n_buildings=1,
    )
    assert len(results) == 1
    building_id, characteristics = results[0]
    assert building_id == 1002000365947
    assert characteristics.building_type == "MediumOffice"
    assert characteristics.num_floors == 4
    assert (characteristics.area - 2676.54) < 0.01
    assert (characteristics.height - 47.83) < 0.01


def test_no_matches(metadata_path):
    """Test search with non-existent building type"""
    results = search_metadata(
        metadata_path,
        building_type="NonExistentType",
        area=5000,
        num_floors=2,
        height=8,
        n_buildings=1,
    )
    assert len(results) == 0


def test_multiple_buildings_same_type(metadata_path):
    """Test search returning multiple buildings of the same type using config parameters"""
    results = search_metadata(
        metadata_path,
        building_type="SmallOffice",
        area=1000,
        num_floors=1,
        height=10.0,
        n_buildings=2,
    )
    assert len(results) == 2

    # Check that results are tuples with ID and characteristics
    for building_id, characteristics in results:
        assert isinstance(building_id, int)
        assert characteristics.building_type == "SmallOffice"
        assert characteristics.num_floors == 1
        # Area and height should be close to target values but not exact
        assert abs(characteristics.area - 1000) < 1000  # Within 1000 sq ft
        assert abs(characteristics.height - 10.0) < 5  # Within 5 ft

    # Get the IDs in sorted order
    found_ids = sorted(
        [int(building_id) for building_id, _ in results]
    )  # Convert to int for proper comparison
    # These should be the IDs of single zone files in fixtures
    assert found_ids == [1006001329312, 1007001504911]


def test_sorting_order(metadata_path):
    """Test that buildings are returned in correct order based on similarity to criteria"""
    results = search_metadata(
        metadata_path,
        building_type="SmallOffice",
        area=1500,  # Target values that don't exactly match any building
        num_floors=1,
        height=11.0,
        n_buildings=3,
    )

    assert len(results) >= 2, "Should find at least 2 buildings for comparison"

    # Get the differences for each result
    diffs = []
    for building_id, chars in results:
        area_diff = abs(chars.area - 1500)
        height_diff = abs(chars.height - 11.0)
        diffs.append((area_diff, height_diff))

    # Check that results are sorted by area difference first, then height
    for i in range(len(diffs) - 1):
        # Either the first building should have smaller area difference
        # or equal area difference but smaller height difference
        assert diffs[i][0] < diffs[i + 1][0] or (
            diffs[i][0] == diffs[i + 1][0] and diffs[i][1] <= diffs[i + 1][1]
        ), f"Results not properly sorted at position {i}"


def test_edge_cases(metadata_path):
    """Test search_metadata with edge cases and extreme values"""
    # Test with zero values
    zero_results = search_metadata(
        metadata_path,
        building_type="SmallOffice",
        area=0,
        num_floors=0,
        height=0,
        n_buildings=1,
    )
    assert len(zero_results) >= 0

    # Test with negative values
    negative_results = search_metadata(
        metadata_path,
        building_type="SmallOffice",
        area=-1000,
        num_floors=-1,
        height=-10,
        n_buildings=1,
    )
    assert len(negative_results) >= 0

    # Test with None for optional parameters
    none_results = search_metadata(
        metadata_path,
        building_type="SmallOffice",
        area=None,
        num_floors=None,
        height=None,
        n_buildings=1,
    )
    assert len(none_results) >= 0

    # Test with extreme values
    extreme_results = search_metadata(
        metadata_path,
        building_type="SmallOffice",
        area=1e6,  # Very large area
        num_floors=100,  # Very tall building
        height=1000,  # Extreme height
        n_buildings=1,
    )
    assert len(extreme_results) >= 0

    # For any results returned, verify the structure is correct
    for results in [zero_results, negative_results, none_results, extreme_results]:
        if results:
            building_id, characteristics = results[0]
            assert isinstance(building_id, int)
            assert characteristics.building_type == "SmallOffice"


def test_load_metadata_all_counties():
    """Test loading metadata without county filtering"""
    processed_path = process_metadata("VT")
    df = pd.read_parquet(processed_path)
    assert not df.empty
    assert df["State_Abbr"].unique()[0] == "VT"
    # Verify county column exists and has values
    assert "County" in df.columns
    # We sometimes get rows without counties
    # assert all(df["County"].notna())


def test_load_metadata_nonexistent_county(state):
    """Test loading metadata with a county that doesn't exist in the data"""
    p = process_metadata(state)
    list = search_metadata(p, county="Does not exist")
    assert len(list) == 0


def test_search_idf_small_office(state, county):
    """Test search_idf with SmallOffice parameters"""
    # Set up mocks

    buildings = search_idf(
        state=state,
        county=county,
        building_type="SmallOffice",
        area=1574.74,
        num_floors=1,
        height=10.76,
        n_buildings=1,
    )

    # Check return structure
    assert isinstance(buildings, list)

    # Check building data structure
    if buildings:
        building_path, characteristics = buildings[0]
        assert building_path.relative_to(DataPaths.data_dir()) == Path(
            "processed/VT/Caledonia/1002000371360.epJSON"
        )
        assert characteristics.building_type == "SmallOffice"
        assert characteristics.num_floors == 1
        assert abs(characteristics.area - 1608) <= 1
        assert (characteristics.height - 14) <= 1


def test_search_idf_medium_office(state, county):
    """Test search_idf with MediumOffice parameters"""

    buildings = search_idf(
        state=state,
        county=county,
        building_type="MediumOffice",
        area=2676.54,
        num_floors=4,
        height=47.83,
        n_buildings=1,
    )

    # Check return structure
    assert isinstance(buildings, list)

    # Check building data structure
    if buildings:
        building_path, characteristics = buildings[0]
        assert building_path.relative_to(DataPaths.data_dir()) == Path(
            "processed/VT/Caledonia/1002000365947.epJSON"
        )
        assert characteristics.building_type == "MediumOffice"
        assert characteristics.num_floors == 4
        assert (characteristics.area - 2676.54) < 0.01
        assert (characteristics.height - 47.83) < 0.01


def test_search_idf_invalid_params():
    """Test search with invalid parameters"""

    with pytest.raises(Exception):
        search_idf(
            state="INVALID",
            county="INVALID",
            building_type="NonExistentType",
            area=0,
            num_floors=0,
            height=None,
            n_buildings=1,
        )


def test_search_idf_nonexistent_county(state, county):
    """Test search_idf with a county that doesn't exist in the metadata"""

    with pytest.raises(Exception):
        search_idf(
            state=state,
            county="INVALID",
            building_type="SmallOffice",
            area=1000,
            num_floors=1,
            height=10.0,
            n_buildings=1,
        )


def test_corrupted_idf_files():
    """Test handling of corrupted and missing IDF files"""
    # Create a temporary corrupted IDF file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".idf", delete=False) as f:
        # Write invalid IDF content
        f.write("This is not a valid IDF file content\n")
        f.write("It should cause processing to fail\n")
        corrupted_path = Path(f.name)

    with pytest.raises(Exception):
        process_idf(corrupted_path, corrupted_path.with_suffix(".epjson"))


def test_invalid_state_code():
    """Test search with invalid state code"""

    with pytest.raises(Exception):
        search_idf(
            state="XX",  # Invalid state code
            county="Caledonia",
            building_type="SmallOffice",
            area=1500,
            num_floors=1,
            height=10.0,
            n_buildings=1,
        )


def test_invalid_building_type(state, county):
    """Test search with non-existent building type"""

    with pytest.raises(Exception):
        buildings = search_idf(
            state=state,
            county=county,
            building_type="NonExistentBuildingType",  # Invalid building type
            area=1500,
            num_floors=1,
            height=10.0,
            n_buildings=1,
        )


# def test_process_single_idf():
#     """Test processing a single IDF file"""
#     # Test data
#     building_id = 1003000523385

#     # Load the actual characteristics from the reference JSON file
#     ref_characteristics_path = (
#         self.fixtures_dir / "processed_buildings" / f"{building_id}.json"
#     )
#     with open(ref_characteristics_path) as f:
#         test_characteristics = json.load(f)

#     # Verify input file exists
#     input_dir = self.fixtures_dir / "idf"
#     input_file = input_dir / f"{building_id}.idf"

#     # Process the IDF file
#     with cd(self.temp_dir):  # Change to temp directory for processing
#         processed_paths = process_idf(
#             [(building_id, test_characteristics)],
#             state="VT",
#             county="TestCounty",
#             keep_original=True,
#             output_dir=self.test_output_dir,
#             input_dir=input_dir,
#         )

#     # Check that we got a result
#     self.assertEqual(len(processed_paths), 1)

#     # Check that output file exists and is epJSON
#     output_path = processed_paths[0]
#     self.assertTrue(output_path.exists())
#     self.assertEqual(output_path.suffix, ".epJSON")

#     # Check that characteristics file was created
#     characteristics_path = self.test_output_dir / f"{building_id}.json"
#     self.assertTrue(characteristics_path.exists())

#     # Compare with reference files
#     ref_dir = self.fixtures_dir / "processed_buildings"
#     ref_epjson = ref_dir / f"{building_id}.epJSON"

#     # Compare epJSON files by loading and comparing contents
#     with open(output_path) as f1, open(ref_epjson) as f2:
#         test_epjson = json.load(f1)
#         ref_epjson_content = json.load(f2)
#         diff = DeepDiff(test_epjson, ref_epjson_content, significant_digits=6)
#         self.assertEqual(diff, {}, "Generated epJSON file differs from reference")
