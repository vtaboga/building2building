import os
import subprocess
import pytest
import shutil
from pathlib import Path
import time

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="function")
def output_dir():
    """Fixture to provide and cleanup output directory"""
    project_root = get_project_root()
    output_dir = project_root / "tests" / "output"
    output_dir.mkdir(exist_ok=True)
    
    yield output_dir
    
    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)

def test_energyplus_simulation(output_dir):
    """Test that EnergyPlus can run a simulation successfully in the container."""
    project_root = get_project_root()
    idf_path = project_root / "tests" / "fixtures" / "small_office.idf"
    weather_path = project_root / "tests" / "fixtures" / "weather_small_office.epw"
    container_path = project_root / "energyplus" / "singularity_eplus.sif"
    
    # Verify container exists
    assert container_path.exists(), "Singularity container not found"
    
    # Run EnergyPlus simulation
    cmd = [
        "singularity", "exec",
        "-B", f"{idf_path.parent}:/input",
        "-B", f"{output_dir}:/output",
        "--pwd", "/input",
        str(container_path),
        "/usr/local/EnergyPlus-24-1-0/energyplus",
        "-w", "/input/weather_small_office.epw",
        "-d", "/output",
        "-r",
        "small_office.idf"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Wait for files to be written
        time.sleep(15)
        
        # Verify simulation completed successfully
        assert result.returncode == 0, "EnergyPlus simulation failed"
        assert "EnergyPlus Completed Successfully" in result.stderr, "EnergyPlus did not complete successfully"
        
        # Check for expected output files
        expected_files = ["eplusout.err", "eplusout.rdd"]
        missing_files = []
        for file in expected_files:
            if not (output_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            output_files = list(output_dir.glob('*'))
            pytest.fail(f"Missing expected output files: {', '.join(missing_files)}\n"
                       f"Found files: {', '.join(f.name for f in output_files)}")
            
    except subprocess.CalledProcessError as e:
        pytest.fail(f"EnergyPlus simulation failed with error:\n"
                   f"STDOUT:\n{e.stdout}\n"
                   f"STDERR:\n{e.stderr}\n"
                   f"Return code: {e.returncode}") 
