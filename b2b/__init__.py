from b2b.env import setup_energyplus_path

setup_energyplus_path()

from gymnasium.envs.registration import register

register(
    id="EnergyPlus-v0",
    entry_point="b2b.simulator:create_simulator",
    kwargs={
        "building_config": None,  # Will be provided when creating env
    },
)
