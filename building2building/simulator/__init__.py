"""
Simulator package initialization.
Ensures EnergyPlus path is set up before any imports that depend on it.
"""

# Register the environment with Gymnasium
from gymnasium.envs.registration import register

register(
    id="EnergyPlus-v0",
    entry_point="building2building.simulator.create_simulator:create_simulator",
    kwargs={
        "building_config": None,  # Will be provided when creating env
    },
)
