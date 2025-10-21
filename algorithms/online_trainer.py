import importlib
import inspect
from pathlib import Path
from omegaconf import OmegaConf
from wandb.integration.sb3 import WandbCallback

from algorithms.utils import make_dummy_vec_env, make_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


def _make_envs(config: OmegaConf, output_dir: Path):
    num_envs = config.training.get('num_train_envs')
    train_env = make_vec_env(
        make_env,
        n_envs=int(num_envs),
        env_kwargs={'eplus_output_dir': str(output_dir / "train_eplus_outputs")},
    )
    eval_env = make_dummy_vec_env(
        eplus_output_dir=str(output_dir / "eval_eplus_outputs")
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

