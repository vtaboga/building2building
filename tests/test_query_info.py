import rdflib
import os
import json
import sys

# Add EnergyPlus path to system path
sys.path.append("/usr/local/EnergyPlus-22-1-0")

import src.simulator.query_info as query_info
from src.simulator.config import get_controllable_setpoints_rdf


def test_get_controllable_setpoints():
    # Load an existing epJSON file from the data directory
    epjson_path = os.path.join('tests', 'fixtures', 'building.epJSON')
    rdf = query_info.rdf_from_json(epjson_path)
    
    # Get controllable setpoints
    setpoints = get_controllable_setpoints_rdf(rdf)
    
    # Test that we only have setpoints for Space 1 ZN
    assert list(setpoints.keys()) == ['Space 1 ZN'], "Only Space 1 ZN should have setpoints"
    
    # Get the controls for Space 1 ZN
    controls = setpoints['Space 1 ZN']
    
    # Test that we have both heating and cooling setpoints
    assert len(controls) == 2, "Should have both heating and cooling setpoints"
    
    # Sort controls by type to ensure consistent order
    controls = sorted(controls, key=lambda x: x['setpoint_type'])
    
    # Test cooling setpoint
    cooling = controls[0]
    assert cooling['setpoint_type'] == 'cooling'
    assert cooling['schedule_name'] == 'MidriseApartment Apartment ClgSetp'
    assert cooling['control_type'] == 'DualSetpoint'
    assert cooling['actuator_key'] == 'Zone Temperature Control,Temperature Cooling Setpoint,Space 1 ZN'
    
    # Test heating setpoint
    heating = controls[1]
    assert heating['setpoint_type'] == 'heating'
    assert heating['schedule_name'] == 'MidriseApartment Apartment HtgSetp'
    assert heating['control_type'] == 'DualSetpoint'
    assert heating['actuator_key'] == 'Zone Temperature Control,Temperature Heating Setpoint,Space 1 ZN'


def test_rdf_zones():
    # Load the test building
    epjson_path = os.path.join('tests', 'fixtures', 'building.epJSON')
    rdf = query_info.rdf_from_json(epjson_path)
    
    # Get zones
    zones = query_info.rdf_zones(rdf)
    
    # Test that we have both expected zones
    expected_zones = ['Space 0 ZN', 'Space 1 ZN']
    assert sorted(zones) == sorted(expected_zones), "Should find both zones"


def test_rdf_schedules():
    # Load the test building
    epjson_path = os.path.join('tests', 'fixtures', 'building.epJSON')
    rdf = query_info.rdf_from_json(epjson_path)
    
    # Get schedules
    schedules = query_info.rdf_schedules(rdf)
    
    # Test that we have the expected heating and cooling schedules
    assert 'MidriseApartment Apartment ClgSetp' in schedules, "Cooling schedule not found"
    assert 'MidriseApartment Apartment HtgSetp' in schedules, "Heating schedule not found"






