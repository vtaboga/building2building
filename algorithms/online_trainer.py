import importlib
import inspect
import logging
from pathlib import Path
from omegaconf import OmegaConf
from wandb.integration.sb3 import WandbCallback
import wandb

from algorithms.utils import make_dummy_vec_env, make_env, log_test_dir_graphs_wandb
from algorithms.test import test_policy
from building2building.simulator.wrappers import NormalizeObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed


logger = logging.getLogger(__name__)


def _make_envs(config: OmegaConf, output_dir: Path):

    # Read required config without providing defaults; raise if missing
    norm_obs = config.env.normalize_obs  # expect under env

    def wrapper_fn(env):
        if norm_obs:
            env = NormalizeObservation(env)
        return env

    num_envs = int(config.training.num_train_envs)
    train_env = make_vec_env(
        make_env,
        n_envs=num_envs,
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
        gradient_save_freq=config.training.cb_gradient_save_freq,
        model_save_path=str(model_dir),
        verbose=2,
    )
    eval_cb = EvalCallback(
        eval_env,
        log_path=str(log_dir),
        eval_freq=config.training.eval_freq,
        best_model_save_path=str(model_dir),
        n_eval_episodes=config.training.eval_episodes,
        deterministic=True,
    )
    return CallbackList([eval_cb, wandb_cb])


def _build_sb3_model(config: OmegaConf, train_env, tb_dir: Path):
    """Build a Stable Baselines 3 model based on the config."""

    params = OmegaConf.to_container(config.policy, resolve=True)
    algo_name = config.policy.algorithm

    if not algo_name:
        raise ValueError("The key algorithm in policy config is missing.")

    # Resolve algorithm class from SB3 or SB3-Contrib (e.g. TRPO).
    algo_upper = str(algo_name).upper()
    module = None
    algo_cls = None

    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algo_name}.{algo_name}")
            algo_cls = getattr(module, algo_upper)
            break
        except ModuleNotFoundError:
            continue
        except AttributeError:
            continue

    if algo_cls is None:
        raise ValueError(
            f"Algorithm '{algo_name}' not found. "
            "Expected it to exist as "
            f"stable_baselines3.{algo_name}.{algo_name}.{algo_upper} "
            f"or sb3_contrib.{algo_name}.{algo_name}.{algo_upper}."
        )

    # pass only valid arguments
    sig = inspect.signature(algo_cls.__init__)
    valid_params = {k for k in sig.parameters.keys() if k not in ['self', 'env', 'policy']}
    kwargs = {k: v for k, v in params.items() if k in valid_params}

    kwargs['tensorboard_log'] = str(tb_dir)
    policy = config.policy.policy_type

    model = algo_cls(policy, train_env, **kwargs)

    return model


def _load_best_model(config: OmegaConf, model_dir: Path):
    """Load the best saved SB3 model if available, else return None."""
    algo_name = config.policy.algorithm
    best_path = model_dir / "best_model.zip"
    if not best_path.exists() or not algo_name:
        return None
    algo_upper = str(algo_name).upper()
    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algo_name}.{algo_name}")
            algo_cls = getattr(module, algo_upper)
            return algo_cls.load(str(best_path))
        except Exception:
            continue
    return None


def online_trainer(config: OmegaConf, output_dir: Path):

    # Stable reference to the repository root (avoid relying on Hydra's runtime cwd).
    # File is at: <repo_root>/algorithms/online_trainer.py
    repo_root = Path(__file__).resolve().parents[1]

    wandb_run = wandb.init(
        project=config.wandb.project,
		entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True),
		sync_tensorboard=True,
        dir=str(output_dir),
        save_code=True,
    )

    # Ensure the run is explicitly associated with this code checkout in W&B,
    # even if Hydra changes the runtime working directory.
    if wandb_run is not None:
        try:
            wandb_run.log_code(root=str(repo_root))
        except Exception as e:
            logger.warning("wandb log_code failed: %s", e)


    # Prepare IO dirs
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    tb_dir = output_dir / "tb"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    set_random_seed(config.seed)

    # Envs and callbacks
    train_env, eval_env = _make_envs(config, output_dir)
    callbacks = _make_callbacks(config, eval_env, model_dir, log_dir)

    # Build model based on config
    model = _build_sb3_model(config, train_env, tb_dir)

    # Train
    model.learn(
        total_timesteps=int(config.training.total_timesteps),
        callback=[callbacks],
    )

    # Test the saved/best policy for a few episodes
    test_cfg = config
    policy_model = _load_best_model(config, model_dir) or model
    test_policy(test_cfg, policy_model, output_dir)

    log_test_dir_graphs_wandb(output_dir / "test")


    if wandb_run is not None:
        wandb_run.finish()

