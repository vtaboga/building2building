from pathlib import Path
import uuid
from building2building.simulator import create_simulator
from building2building.sources import hydroquebec
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(eplus_output_dir: str):
    # EnergyPlus needs a unique output dir for each run
    eplus_output_dir = Path(eplus_output_dir) / str(uuid.uuid4())
    config = hydroquebec.search_config(eplus_output_dir=str(eplus_output_dir))
    env = create_simulator(config)
    return env


def make_dummy_vec_env(eplus_output_dir: str, seed: int | None = None, wrapper_fn=None):
    """Create a single-env DummyVecEnv with an optional wrapper_fn applied."""
    def _thunk():
        # seed is accepted for API compatibility but not used here
        env = make_env(eplus_output_dir=eplus_output_dir)
        env = Monitor(env)
        if wrapper_fn is not None:
            env = wrapper_fn(env)
        return env
    return DummyVecEnv([_thunk])

