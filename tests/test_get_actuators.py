import json
from pathlib import Path
from minergym.ontology import Ontology
from building2building.simulator.action_spaces import get_hvac_actuators

def test_get_hvac_actuators_with_hydroquebec():
    """Test with real building from hydroquebec dataset"""
    from building2building.sources import hydroquebec
    
    config = hydroquebec.search_configs({}, n=1)[0]
    ont = Ontology.from_json(config.path_to_building)
    
    actuators = get_hvac_actuators(ont)

    print("ACTUATORS:")

    print(actuators)
    
    # Verify structure
    assert isinstance(actuators, dict)