import logging
import pdb
import sys
import json

from pathlib import Path
from b2b.env import STORE_PATH, energyplus_path
from b2b.simulator import create_simulator
from b2b.pipeline.actuators import make_controllable
from b2b.store import realize, LocalFile, Constant

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


