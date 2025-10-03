from building2building.env import setup_energyplus_path

setup_energyplus_path()

from gymnasium.envs.registration import register

register(
    id="EnergyPlus-v0",
    entry_point="building2building.simulator:create_simulator",
    kwargs={
        "building_config": None,  # Will be provided when creating env
    },
)
