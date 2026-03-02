from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def _derived_wandb_tags(cfg: Any) -> list[str]:
    """Add lightweight tags for experiment filtering."""
    tags: list[str] = []

    algo = getattr(getattr(cfg, "policy", None), "algorithm", None)
    if isinstance(algo, str) and algo.strip():
        tags.append(f"algo:{algo.strip().lower()}")

    policy_type = getattr(getattr(cfg, "policy", None), "type", None)
    if isinstance(policy_type, str) and policy_type.strip():
        tags.append(f"policy:{policy_type.strip().lower()}")

    split = getattr(getattr(cfg, "benchmark", None), "split", None)
    if isinstance(split, str) and split.strip():
        tags.append(f"split:{split.strip().lower()}")

    bldg = getattr(cfg, "bldg", None)
    if bldg is not None:
        bldg_inner = getattr(bldg, "bldg", None)
        if bldg_inner is not None:
            bt = getattr(bldg_inner, "building_type", None)
            if isinstance(bt, str) and bt.strip():
                tags.append(f"building:{bt.strip()}")
        sel = getattr(bldg, "selection", None)
        if sel is not None:
            sel_split = getattr(sel, "split", None)
            if isinstance(sel_split, str) and sel_split.strip():
                tags.append(f"split:{sel_split.strip().lower()}")

    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def derive_wandb_run_name(cfg: Any, env_metadata: dict[str, Any]) -> str | None:
    """Build a descriptive run name with a short unique suffix."""
    import uuid

    parts: list[str] = []

    source = env_metadata.get("building_source_metadata", {})
    bt = source.get("building_type") or source.get("source", "")
    if bt:
        parts.append(str(bt))

    bid = source.get("building_id")
    if bid is not None:
        parts.append(f"id{bid}")

    policy_type = getattr(getattr(cfg, "policy", None), "type", None)
    algo = getattr(getattr(cfg, "policy", None), "algorithm", None)
    label = algo or policy_type
    if isinstance(label, str) and label.strip():
        parts.append(label.strip())

    parts.append(uuid.uuid4().hex[:6])

    return "_".join(parts)


def _extract_wandb_config(cfg: Any) -> dict[str, Any]:
    """Extract only the fields useful as wandb run columns."""
    try:
        full = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        full = cfg if isinstance(cfg, dict) else {}
    if not isinstance(full, dict):
        full = {}

    out: dict[str, Any] = {}

    policy = full.get("policy", {})
    if isinstance(policy, dict):
        for k in ("type", "algorithm"):
            if k in policy:
                out[f"policy/{k}"] = policy[k]

    reward = full.get("reward", {})
    if isinstance(reward, dict):
        for k in ("reward_type", "energy_weight", "dT", "deadband_c"):
            if k in reward:
                out[f"reward/{k}"] = reward[k]

    task = full.get("task", {})
    if isinstance(task, dict):
        if "run_period" in task:
            out["task/run_period"] = task["run_period"]
        if "target_temperature_mode" in task:
            out["task/target_temperature_mode"] = task["target_temperature_mode"]

    bldg = full.get("bldg", {})
    if isinstance(bldg, dict):
        inner = bldg.get("bldg", {})
        if isinstance(inner, dict) and "building_type" in inner:
            out["building_type"] = inner["building_type"]
        sel = bldg.get("selection", {})
        if isinstance(sel, dict):
            for k in ("split", "index"):
                if k in sel:
                    out[f"selection/{k}"] = sel[k]

    env = full.get("env", {})
    if isinstance(env, dict):
        if "max_steps" in env:
            out["env/max_steps"] = env["max_steps"]

    return out


def init_wandb_from_config(
    cfg: Any,
    *,
    run_dir: Path,
    save_code: bool = True,
    log_code_root: Path | None = None,
    sync_tensorboard: bool | None = None,
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

    enabled = getattr(wandb_cfg, "enabled", True)
    if not enabled:
        return None, False

    project = getattr(wandb_cfg, "project", None)
    if project is None:
        return None, False

    entity = getattr(wandb_cfg, "entity", None)
    tags = getattr(wandb_cfg, "tags", None)
    user_tags = list(tags) if tags is not None else []
    extra_tags = _derived_wandb_tags(cfg)
    merged_tags = [str(t) for t in (user_tags + extra_tags) if str(t).strip()]

    wandb_config = _extract_wandb_config(cfg)

    init_kwargs: dict[str, Any] = {
        "project": str(project),
        "entity": str(entity) if entity is not None else None,
        "tags": merged_tags,
        "config": wandb_config,
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


_MAX_PLOT_POINTS = 2000


def _short_label(col: str) -> str:
    """Strip obs::/act:: prefixes and truncate for chart readability."""
    label = col.removeprefix("obs::").removeprefix("act::")
    if len(label) > 40:
        label = label[:37] + "..."
    return label


def wandb_log_df_line_series(
    *,
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    key_prefix: str,
    title: str,
) -> None:
    """Log a multi-series line plot to wandb as a chart panel.

    Series labels are shortened so they render cleanly regardless of
    how many zones or actuators the building has.
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
        sub = df[[x] + y_cols]
        if len(sub) > _MAX_PLOT_POINTS:
            step = max(1, len(sub) // _MAX_PLOT_POINTS)
            sub = sub.iloc[::step]

        xs = sub[x].to_list()
        keys = [_short_label(c) for c in y_cols]
        ys = [sub[c].to_list() for c in y_cols]
        chart = wandb.plot.line_series(xs, ys, keys=keys, title=title, xname=x)
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
    """Log a single-series line plot (x,y) to W&B as a chart panel."""
    try:
        import wandb  # type: ignore
    except Exception:
        return

    if wandb.run is None:
        return

    try:
        x_list = list(x)
        y_list = [float(v) for v in y]
        if len(x_list) > _MAX_PLOT_POINTS:
            step = max(1, len(x_list) // _MAX_PLOT_POINTS)
            x_list = x_list[::step]
            y_list = y_list[::step]

        chart = wandb.plot.line_series(
            x_list, [y_list], keys=[y_name], title=title, xname=x_name
        )
        wandb.log({key: chart})
    except Exception as e:
        logger.warning("wandb logging failed for %s: %s", key, e)

