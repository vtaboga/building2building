"""
Simulator package initialization.
Ensures EnergyPlus path is set up before any imports that depend on it.
"""
# Register the environment with Gymnasium
from gymnasium.envs.registration import register

register(
    id='EnergyPlus-v0',
    entry_point='building2building.simulator.create_simulator:create_simulator',
    kwargs={
        'path_to_building': None,  # Will be provided when creating env
        'path_to_weather': None,   # Will be provided when creating env
        'building_characteristics': None,
        'reward_type': None,
        'energy_weight': None,
        'eplus_output_dir': None,
    }
)
