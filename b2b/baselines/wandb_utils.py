from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def _derived_wandb_tags(cfg: Any) -> list[str]:
    """
    Add lightweight tags for experiment filtering.

    We keep this intentionally simple: the RL method and the data split.
    """
    tags: list[str] = []

    # RL algorithm (SB3): cfg.policy.algorithm (e.g. "sac", "ppo")
    algo = getattr(getattr(cfg, "policy", None), "algorithm", None)
    if isinstance(algo, str) and algo.strip():
        tags.append(f"algo:{algo.strip().lower()}")

    # Baseline controllers: cfg.policy.type
    policy_type = getattr(getattr(cfg, "policy", None), "type", None)
    if isinstance(policy_type, str) and policy_type.strip():
        tags.append(f"policy:{policy_type.strip().lower()}")

    # Dataset split (you run RL with benchmark.split=test)
    split = getattr(getattr(cfg, "benchmark", None), "split", None)
    if isinstance(split, str) and split.strip():
        tags.append(f"split:{split.strip().lower()}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def init_wandb_from_config(
    cfg: Any,
    *,
    run_dir: Path,
    save_code: bool = True,
    log_code_root: Path | None = None,
    sync_tensorboard: bool | None = None,
    extra_tags: Sequence[str] | None = None,
) -> tuple[object | None, bool]:
    """
    Initialize a W&B run if `cfg` contains a `wandb` section.

    - If a run already exists (`wandb.run is not None`), returns it and `False`.
    - If W&B isn't available or config is missing, returns (None, False).
    - If init succeeds, returns (run, True).
    """
    try:
        import wandb  # type: ignore
    except Exception as e:
        logger.info("wandb not available; skipping logging (%s)", e)
        return None, False

    if getattr(wandb, "run", None) is not None:
        return wandb.run, False

    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None:
        return None, False

    project = getattr(wandb_cfg, "project", None)
    if project is None:
        return None, False

    entity = getattr(wandb_cfg, "entity", None)
    tags = getattr(wandb_cfg, "tags", None)
    user_tags = list(tags) if tags is not None else []
    derived_tags = _derived_wandb_tags(cfg)
    extra_tags_list = list(extra_tags) if extra_tags else []
    merged_tags = [str(t) for t in (user_tags + derived_tags + extra_tags_list) if str(t).strip()]

    # `cfg` is typically an OmegaConf/DictConfig, but keep this helper usable
    # with plain dicts / objects in scripts.
    try:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_dict = cfg if isinstance(cfg, dict) else {}
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    init_kwargs: dict[str, Any] = {
        "project": str(project),
        "entity": str(entity) if entity is not None else None,
        "tags": merged_tags,
        "config": cfg_dict,
        "dir": str(run_dir),
        "save_code": save_code,
    }
    if sync_tensorboard is not None:
        init_kwargs["sync_tensorboard"] = bool(sync_tensorboard)

    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        logger.warning("wandb.init failed; skipping logging: %s", e)
        return None, False

    if run is not None and log_code_root is not None:
        try:
            run.log_code(root=str(log_code_root))
        except Exception as e:
            logger.warning("wandb log_code failed: %s", e)

    return run, True


def finish_wandb_if_started(wandb_run: object | None, *, started_here: bool) -> None:
    if not started_here or wandb_run is None:
        return
    try:
        # wandb_run is a wandb.sdk.wandb_run.Run but keep dependency soft here.
        wandb_run.finish()  # type: ignore[attr-defined]
    except Exception:
        pass


def wandb_log_df_line_series(
    *,
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    key_prefix: str,
    title: str,
) -> None:
    """
    Log an interactive multi-series line plot.

    Note: we intentionally do NOT log W&B Tables to avoid cluttering the run UI.
    Raw rollout data is already persisted to disk as CSV/NPZ by the caller.
    """
    try:
        import wandb  # type: ignore
    except Exception:
        return

    if wandb.run is None:
        return

    y_cols = [c for c in y_cols if c in df.columns]
    if not y_cols or x not in df.columns:
        return

    try:
        # Newer W&B (e.g. 0.23.x): line_series takes raw x/ys, not a Table.
        #
        # signature (as of 0.23.1):
        #   line_series(xs, ys, keys=None, title="", xname="x", ...)
        xs = df[x].to_list()
        ys = [df[c].to_list() for c in y_cols]
        chart = wandb.plot.line_series(xs, ys, keys=y_cols, title=title, xname=x)

        # Log plot under a key that naturally groups into a W&B panel.
        # E.g. key_prefix="rollout/temperature" appears under the "rollout" panel.
        wandb.log({str(key_prefix): chart})
    except Exception as e:
        logger.warning("wandb logging failed for %s: %s", key_prefix, e)


def wandb_log_xy_series(
    *,
    x: Sequence[int] | Sequence[float],
    y: Sequence[float],
    key: str,
    title: str,
    x_name: str = "timestep",
    y_name: str = "value",
) -> None:
    """Log a single-series line plot (x,y) to W&B."""
    try:
        import wandb  # type: ignore
    except Exception:
        return

    if wandb.run is None:
        return

    try:
        table = wandb.Table(columns=[x_name, y_name])
        for xi, yi in zip(x, y):
            table.add_data(xi, float(yi))
        chart = wandb.plot.line(table, x=x_name, y=y_name, title=title)
        wandb.log({key: chart})
    except Exception as e:
        logger.warning("wandb logging failed for %s: %s", key, e)
