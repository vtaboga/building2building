from b2b import simulator
from b2b.env import STORE_PATH, binaries
from b2b.sources import hydroquebec
from b2b.store import realize

import pytest

pytestmark = pytest.mark.long


def test_binary():
    realize(STORE_PATH.get(), binaries["linux-x86_64"])


def test_hydroquebec():
    hydroquebec.search_configs({"env": {"control_mode": "hvac_actuators"}}, n=1)


def test_hydroquebec_env():
    config = hydroquebec.search_configs({"env": {"control_mode": "hvac_actuators"}}, n=1)[0]

    env = simulator.create_simulator(config)

    env.reset()
    stop = truncate = False

    while not (stop or truncate):
        _, _, stop, truncate, _ = env.step(env.action_space.sample())
