import logging
import pdb
import sys
import json

from pathlib import Path
from building2building.env import STORE_PATH, energyplus_path
from building2building.simulator import create_simulator
from building2building.pipeline.actuators import make_controllable
from building2building.store import realize, LocalFile, Constant

logging.basicConfig(level=logging.INFO)

small_office_epjson_path = "./data/SmallOffice_ASHRAE.epjson"

with open(small_office_epjson_path, 'r') as f:
    epjson_data = json.load(f)

EPJSON_PATH = Constant(small_office_epjson_path)
EP_PATH = energyplus_path()

# Create the discovery pipeline for the small office IDF
control_expr = make_controllable(EPJSON_PATH)

control_epjson, actuator_descriptions = realize(STORE_PATH.get(), control_expr)

print("Actuator descriptions: ", actuator_descriptions)


