import importlib
import inspect
from pathlib import Path
from omegaconf import OmegaConf
from wandb.integration.sb3 import WandbCallback

from algorithms.utils import make_dummy_vec_env, make_env
from algorithms.test import test_policy
from building2building.simulator.wrappers import NormalizeObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


def _make_envs(config: OmegaConf, output_dir: Path):

    norm_obs = OmegaConf.select(config, "env.normalize_obs")

    def wrapper_fn(env):
        if norm_obs:
            env = NormalizeObservation(env)
        return env

    num_envs = config.training.get('num_train_envs')
    train_env = make_vec_env(
        make_env,
        n_envs=int(num_envs),
        env_kwargs={'config': config,'eplus_output_dir': str(output_dir / "train_eplus_outputs")},
        wrapper_class=wrapper_fn,  
    )
    eval_env = make_dummy_vec_env(
        config=config,
        eplus_output_dir=str(output_dir / "eval_eplus_outputs"),
        wrapper_fn=wrapper_fn,  
    )
    return train_env, eval_env


def _make_callbacks(config: OmegaConf, eval_env, model_dir: Path, log_dir: Path):
    wandb_cb = WandbCallback(
        gradient_save_freq=config.training.get('cb_gradient_save_freq'),
        model_save_path=str(model_dir),
        verbose=2,
    )
    eval_cb = EvalCallback(
        eval_env,
        log_path=str(log_dir),
        eval_freq=config.training.get('eval_freq'),
        best_model_save_path=str(model_dir),
        n_eval_episodes=config.training.get('eval_episodes'),
        deterministic=True,
    )
    return CallbackList([eval_cb, wandb_cb])


def _build_sb3_model(config: OmegaConf, train_env, tb_dir: Path):
    """Build a Stable Baselines 3 model based on the config."""

    params = OmegaConf.to_container(config.policy, resolve=True)
    algo_name = config.policy.get("algorithm")

    if not algo_name:
        raise ValueError("The key algorithm in policy config is missing.")

    try:
        # Load the policy class from stable_baselines3 according to the name
        module_name = f"stable_baselines3.{algo_name}"
        module = importlib.import_module(f"{module_name}.{algo_name}")
        algo_cls = getattr(module, algo_name.upper())  

    except ModuleNotFoundError as e:
        raise ValueError(f"Algorithm module '{module_name}' not found in stable baselines 3.") from e
    except AttributeError as e:
        raise ValueError(f"Algorithm class '{algo_name}' not found in module '{module_name}'.") from e

    # pass only valid arguments
    sig = inspect.signature(algo_cls.__init__)
    valid_params = {k for k in sig.parameters.keys() if k not in ['self', 'env', 'policy']}
    kwargs = {k: v for k, v in params.items() if k in valid_params}

    kwargs['tensorboard_log'] = str(tb_dir)
    policy = config.policy.get('policy_type')

    model = algo_cls(policy, train_env, **kwargs)

    return model


def _load_best_model(config: OmegaConf, model_dir: Path):
    """Load the best saved SB3 model if available, else return None."""
    algo_name = config.policy.get("algorithm")
    best_path = model_dir / "best_model.zip"
    if not best_path.exists() or not algo_name:
        return None
    try:
        module = importlib.import_module(f"stable_baselines3.{algo_name}.{algo_name}")
        algo_cls = getattr(module, algo_name.upper())
        return algo_cls.load(str(best_path))
    except Exception:
        return None


def online_trainer(config: OmegaConf, output_dir: Path, wandb_run):
    # Prepare IO dirs
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    tb_dir = output_dir / "tb"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Envs and callbacks
    train_env, eval_env = _make_envs(config, output_dir)
    callbacks = _make_callbacks(config, eval_env, model_dir, log_dir)

    # Build model based on config
    model = _build_sb3_model(config, train_env, tb_dir)

    # Train
    model.learn(
        total_timesteps=int(config.training.get('total_timesteps')),
        callback=[callbacks],
    )

    if wandb_run is not None:
        wandb_run.finish()

    # Test the saved/best policy for a few episodes
    test_cfg = config
    # ensure a default if not present (test.yaml sets this normally)
    if not hasattr(test_cfg, "n_episodes"):
        OmegaConf.set_struct(test_cfg, False)
        test_cfg.n_episodes = 1
    policy_model = _load_best_model(config, model_dir) or model
    test_policy(test_cfg, policy_model, output_dir)

