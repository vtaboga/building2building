from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from building2building.baselines.wandb_utils import finish_wandb_if_started, init_wandb_from_config
from building2building.benchmark.problem_adaptive_dynamics import AdaptiveDynamicsProblem

logger = logging.getLogger(__name__)


def make_policy_from_cfg(cfg: DictConfig) -> Any:
    policy_type = str(getattr(cfg.policy, "type", "")).strip()
    if policy_type == "unitary_g36":
        from building2building.baselines.controllers.unitary_g36 import (  # noqa: WPS433
            UnitaryG36Policy,
        )

        return UnitaryG36Policy(cfg.policy)
    if policy_type == "ashrae_air_loop":
        from building2building.baselines.controllers.ashrae_air_loop import (  # noqa: WPS433
            AshraeAirLoopPolicy,
        )

        return AshraeAirLoopPolicy(cfg.policy)

    raise NotImplementedError(
        f"Unsupported policy.type={policy_type!r} for adaptive dynamics benchmark."
    )


def log_returns_to_wandb(*, returns: list[float]) -> None:
    try:
        import wandb  # type: ignore
    except Exception:
        return

    if getattr(wandb, "run", None) is None:
        return
    if not returns:
        return

    arr = np.asarray(returns, dtype=float)

    # 1) Histogram of returns across evaluated environments.
    try:
        t = wandb.Table(columns=["return"])
        for v in returns:
            t.add_data(float(v))
        wandb.log(
            {
                "adaptive_dynamics/return_hist": wandb.plot.histogram(
                    t, "return", title="Episode return across environments"
                )
            }
        )
    except Exception as e:
        logger.warning("Failed to log return histogram to wandb: %s", e)

    # 2) Aggregate stats across environments as summary scalars + a bar chart.
    stats = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_envs": float(arr.size),
    }

    for k, v in stats.items():
        try:
            wandb.summary[f"adaptive_dynamics/return_{k}"] = v
        except Exception:
            pass

    try:
        t2 = wandb.Table(columns=["stat", "value"])
        for k in ("mean", "median", "std", "min", "max"):
            t2.add_data(k, float(stats[k]))
        wandb.log(
            {
                "adaptive_dynamics/return_stats_bar": wandb.plot.bar(
                    t2,
                    "stat",
                    "value",
                    title="Return stats across environments",
                )
            }
        )
    except Exception as e:
        logger.warning("Failed to log return stats bar chart to wandb: %s", e)


def run_bm_adaptive_dynamics(cfg: DictConfig, *, output_dir: Path) -> int:
    """
    Hydra/W&B wrapper around the adaptive dynamics control problem.

    The core, reusable API is `building2building.benchmark.problem_adaptive_dynamics.AdaptiveDynamicsProblem`.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = make_policy_from_cfg(cfg)

    wandb_run, started_here = init_wandb_from_config(cfg, run_dir=output_dir)
    try:
        cfg_dict_any = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict = cfg_dict_any if isinstance(cfg_dict_any, dict) else {}

        bench = cfg_dict.get("benchmark", {})
        bench_dict: dict[str, object] = bench if isinstance(bench, dict) else {}

        split = str(bench_dict.get("split", "train")).strip().lower()
        if split not in ("train", "test"):
            raise ValueError(f"benchmark.split must be 'train' or 'test', got {split!r}")

        start = int(bench_dict.get("start", 0) or 0)
        limit = int(bench_dict.get("limit", 0) or 0)

        max_steps_raw = bench_dict.get("max_steps", None)
        max_steps: int | None
        try:
            max_steps = int(max_steps_raw) if max_steps_raw is not None else None
        except Exception:
            max_steps = None

        problem = AdaptiveDynamicsProblem(
            split=split,  # type: ignore[arg-type]
            start=start,
            limit=limit,
            max_steps=max_steps,
            base_config=cfg_dict,
        )
        records = problem.run(policy, output_dir=output_dir)
        if wandb_run is not None:
            returns: list[float] = [
                float(r.episode_result.total_reward)
                for r in records
                if r.episode_result is not None
            ]
            log_returns_to_wandb(returns=returns)
    finally:
        finish_wandb_if_started(wandb_run, started_here=started_here)

    return 0

