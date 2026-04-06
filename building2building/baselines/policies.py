"""Policy factory for baseline and user-defined controllers.

Dispatches on a ``"policy.type"`` key in the configuration to load
built-in rule-based controllers, Stable-Baselines3 checkpoints, or
arbitrary custom classes.
"""

from __future__ import annotations

import importlib
from typing import Any

from omegaconf import OmegaConf

from b2b.benchmark.runner import PolicyLike


def _load_sb3_policy(policy_cfg: dict[str, Any]) -> PolicyLike:
    algorithm = str(policy_cfg.get("algorithm", "")).strip()
    checkpoint_path = str(policy_cfg.get("checkpoint_path", "")).strip()
    if not algorithm:
        raise ValueError("policy.algorithm is required for sb3 policies")
    if not checkpoint_path:
        raise ValueError("policy.checkpoint_path is required for sb3 policies")

    path = checkpoint_path
    algo_upper = algorithm.upper()
    for base_pkg in ("stable_baselines3", "sb3_contrib"):
        try:
            module = importlib.import_module(f"{base_pkg}.{algorithm}.{algorithm}")
            algo_cls = getattr(module, algo_upper)
            return algo_cls.load(path)
        except (ModuleNotFoundError, AttributeError):
            continue
    raise ImportError(
        f"SB3 algorithm {algorithm!r} not found in stable_baselines3 or sb3_contrib."
    )


def _load_custom_policy(policy_cfg: dict[str, Any]) -> PolicyLike:
    module_path = str(policy_cfg.get("module", "")).strip()
    class_name = str(policy_cfg.get("class_name", "")).strip()
    if not module_path or not class_name:
        raise ValueError(
            "policy.module and policy.class_name are required for custom policies"
        )
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    kwargs = policy_cfg.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError(f"policy.kwargs must be a mapping, got {type(kwargs).__name__}")
    policy = cls(**kwargs)
    if not hasattr(policy, "predict"):
        raise TypeError(f"{module_path}.{class_name} does not expose a predict() method")
    return policy


def make_policy_from_config(cfg: dict[str, Any]) -> PolicyLike:
    """Instantiate a policy from a Hydra / OmegaConf configuration.

    Supported ``policy.type`` values:

    * ``"unitary_g36"`` / ``"air_loop_sat"`` / ``"ashrae_air_loop"``
      – built-in controllers.
    * ``"sb3"`` – loads a Stable-Baselines3 checkpoint (requires
      ``policy.algorithm`` and ``policy.checkpoint_path``).
    * ``"custom"`` – dynamically imports a user class (requires
      ``policy.module`` and ``policy.class_name``).

    Args:
        cfg: Configuration dict (or OmegaConf) with a ``"policy"``
            sub-section.

    Returns:
        An object satisfying the ``PolicyLike`` protocol (i.e. exposing
        a ``predict`` method).

    Raises:
        NotImplementedError: If the policy type is not supported.
        ValueError: If required keys are missing for the chosen type.
        ImportError: If an SB3 algorithm cannot be found.
    """
    if not isinstance(cfg, dict):
        cfg_any = OmegaConf.to_container(cfg, resolve=True)
        cfg = cfg_any if isinstance(cfg_any, dict) else {}
    policy_cfg = cfg.get("policy", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}
    policy_type = str(policy_cfg.get("type", "")).strip()
    if policy_type == "unitary_g36":
        from b2b.baselines.controllers.unitary_g36 import UnitaryG36Policy

        return UnitaryG36Policy(OmegaConf.create(policy_cfg))
    if policy_type == "air_loop_sat":
        from b2b.baselines.controllers.air_loop_sat import AirLoopSatPolicy

        return AirLoopSatPolicy(OmegaConf.create(policy_cfg))
    if policy_type == "ashrae_air_loop":
        from b2b.baselines.controllers.ashrae_air_loop import AshraeAirLoopPolicy

        return AshraeAirLoopPolicy(OmegaConf.create(policy_cfg))
    if policy_type == "sb3":
        return _load_sb3_policy(policy_cfg)
    if policy_type == "custom":
        return _load_custom_policy(policy_cfg)
    raise NotImplementedError(
        f"Unsupported policy.type={policy_type!r}. "
        "Use built-in baseline names, 'sb3', or 'custom'."
    )
