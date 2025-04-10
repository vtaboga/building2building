from gymnasium.envs.registration import register

register(
    id='EnergyPlus-v0',
    entry_point='src.simulator.create_simulator:create_simulator',
    kwargs={
        'path_to_building': None,  # Will be provided when creating env
        'path_to_weather': None,   # Will be provided when creating env
    }
)
