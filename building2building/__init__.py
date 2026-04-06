try:
    from b2b.env import setup_energyplus_path

    setup_energyplus_path()
except ModuleNotFoundError:
    # Keep import-time side effects optional in lightweight test environments.
    pass

from gymnasium.envs.registration import register

from b2b.api import make_env, make_multizones_env, make_single_zone_env

register(
    id="EnergyPlus-v0",
    entry_point="b2b.simulator:create_simulator",
    kwargs={
        "building_config": None,  # Will be provided when creating env
    },
)

__all__ = ["make_env", "make_single_zone_env", "make_multizones_env"]
