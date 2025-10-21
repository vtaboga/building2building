from pathlib import Path
import uuid
from building2building.simulator import create_simulator
from building2building.sources import hydroquebec
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(eplus_output_dir: str):
    
    eplus_output_dir = Path(eplus_output_dir) / str(uuid.uuid4())  # EnerguPlus needs a unique output dir
    config = hydroquebec.search_config(eplus_output_dir=str(eplus_output_dir))
    env = create_simulator(config)
    return env


def make_dummy_vec_env(eplus_output_dir: str):

    def thunk():
        return Monitor(make_env(eplus_output_dir))

    return DummyVecEnv([thunk])


def test_policy(model, env, n_episodes: int = 1):
    """Test a trained policy."""
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")