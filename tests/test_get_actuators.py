import json
from pathlib import Path
import tempfile
import shutil

from building2building.pipeline import get_hvac_actuators
from building2building.env import STORE_PATH, energyplus_path
from building2building.store import realize


def test_get_hvac_actuators_from_edd():
    """Test parsing existing .edd file"""
    edd_path = Path("tests/fixtures/eplusout.edd")
    
    actuators = get_hvac_actuators(edd_path)
    
    print(f"\nFound {len(actuators)} HVAC actuators:")
    for i, actuator in enumerate(actuators, 1):
        print(f"{i}. {actuator['component_name']}")
        print(f"   Type: {actuator['component_type']}")
        print(f"   Control: {actuator['control_type']}")
        print(f"   Units: {actuator['units']}")
    
    # Verify structure
    assert len(actuators) > 0, "Should find at least some HVAC actuators"
    
    for actuator in actuators:
        assert isinstance(actuator, dict)
        assert "component_name" in actuator
        assert "component_type" in actuator
        assert "control_type" in actuator
        assert "units" in actuator
    
    # Verify no autosized actuators
    for actuator in actuators:
        assert "autosized" not in actuator["control_type"].lower()
        assert "autosized" not in actuator["component_type"].lower()
    
    # Verify expected actuator types are present
    component_types = [a["component_type"] for a in actuators]
    assert any("Coil Speed Control" in ct for ct in component_types)
    assert any("Fan" in ct for ct in component_types)


def test_get_hvac_actuators_with_simulation():
    """Test with bldg1.epjson by running a simulation to generate .edd file"""
    import subprocess
    
    # Get EnergyPlus path
    ep_path = realize(STORE_PATH.get(), energyplus_path())
    
    # Paths to test fixtures
    epjson_path = Path("tests/fixtures/bldg1.epjson")
    epw_path = Path("tests/fixtures/weather.epw")
    
    assert epjson_path.exists(), f"Building file not found: {epjson_path}"
    assert epw_path.exists(), f"Weather file not found: {epw_path}"
    
    # Create temporary directory for simulation outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        print(f"\nRunning EnergyPlus simulation for {epjson_path.name}...")
        
        # Run EnergyPlus simulation
        cmd = [
            str(ep_path / "energyplus"),
            "-d",
            str(tmp_path),
            "-w",
            str(epw_path),
            "-x",  # No need for expandobjects
            str(epjson_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check if simulation succeeded
        edd_file = tmp_path / "eplusout.edd"
        
        if not edd_file.exists():
            print(f"Simulation output:\n{result.stdout}")
            print(f"Simulation errors:\n{result.stderr}")
            raise Exception(f"Simulation did not produce .edd file")
        
        print(f"✓ Simulation completed, .edd file generated")
        
        # Test get_hvac_actuators with the generated .edd file
        actuators = get_hvac_actuators(edd_file)
        
        print(f"\nFound {len(actuators)} HVAC actuators in {epjson_path.name}:")
        for i, actuator in enumerate(actuators, 1):
            print(f"\n{i}. {actuator['component_name']}")
            print(f"   Component Type: {actuator['component_type']}")
            print(f"   Control Type: {actuator['control_type']}")
            print(f"   Units: {actuator['units']}")
        
        # Verify structure
        assert len(actuators) > 0, "Should find at least some HVAC actuators"
        
        for actuator in actuators:
            assert isinstance(actuator, dict)
            assert "component_name" in actuator
            assert "component_type" in actuator
            assert "control_type" in actuator
            assert "units" in actuator
            
            # Verify all values are strings
            assert isinstance(actuator["component_name"], str)
            assert isinstance(actuator["component_type"], str)
            assert isinstance(actuator["control_type"], str)
            assert isinstance(actuator["units"], str)
        
        # Verify no autosized actuators
        for actuator in actuators:
            assert "autosized" not in actuator["control_type"].lower(), \
                f"Found autosized in control_type: {actuator}"
        
        # Verify no schedules are included
        for actuator in actuators:
            assert "Schedule:" not in actuator["component_type"], \
                f"Should not include schedules: {actuator}"
        
        print("\n✓ All assertions passed!")
